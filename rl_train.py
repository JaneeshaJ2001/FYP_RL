"""
rl_train.py  — production-ready for 3,500+ conversation datasets
─────────────────────────────────────────────────────────────────
Train and evaluate the retrieval-decision policy with PPO.

Two-stage training pipeline (PDF: "Supervised warm start"):
  Stage A — Supervised Warm Start
      Train the MLP trunk + action head as a binary classifier using the
      'retrieval_required' labels from the dataset.
      Loss: CrossEntropyLoss(action_logits, retrieval_required)
      Teaches: obvious first-turn retrieval, memory-sufficient skips,
               OOD skips — so PPO doesn't discover these from scratch.

  Stage B — PPO Fine-tuning
      Load Stage A weights (trunk + action head) into PPO's policy.
      Optimize R = judge_score − β × retrieval_cost.
      Value head is learned entirely during PPO.

Usage
─────
  # Full pipeline (warm start → PPO):
    python rl_train.py --episodes conversations.json --timesteps 20000

  # Ablation: skip warm start, start PPO from random weights:
    python rl_train.py --episodes conversations.json --timesteps 20000 --no-warm-start

  # Eval only (load saved policy):
    python rl_train.py --episodes conversations.json --eval-only
    
Key changes vs original:
  1. BUG FIX: warm_start_train() guards best_state_dict=None
  2. BUG FIX: RetrievalStatsCallback accumulates rollout_metrics for plotting
  3. SCALE: --timesteps default lowered to 3000 (realistic for LLM-in-the-loop)
  4. SCALE: --eval-episodes cap (default 50) to keep each eval pass under 5 min
  5. CACHE: warm-start embedding cache via rl_core/embed_cache.py
             (saves ~35s per run after first encode of 16k+ turns)
  6. PLOTS: generate_all_plots() called at end of training
  7. CLI:   --no-plots, --plots-dir, --eval-episodes, --no-cache flags added

Scale guidance (3,500 conversations ≈ 21,000 turns):
  Warm-start encode   : ~35s on CPU  (cached after first run → ~1s)
  Warm-start training : ~30s/epoch × 5–15 epochs ≈ 2–8 min total
  PPO (1 step = 1 LLM turn ≈ 3–5s):
      500 steps  →  ~25-40 min  (smoke test)
    3,000 steps  →  2.5–4 hrs   (recommended minimum)
   20,000 steps  →  17–28 hrs   (full run)
  Evaluation (4 strategies × 50 val eps × 6 turns × 1 judge call):
      ~50 episodes → ~20-30 min per eval pass
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from core.config import CONFIG
from rl_core.evaluation import EvaluationCallback, evaluate
from rl_core.rl_env import RAGDecisionEnv
from rl_core.rl_policy import TransformerHeadPolicy
from rl_core.state_encoder import encode_state
from rl_core.training_trace import configure_training_trace, trace_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_train")

for _noisy in ("stable_baselines3", "httpx", "httpcore", "langfuse", "openai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODES_PATH = (
    REPO_ROOT / "dataset_generator" / "dialog_dataset" / "conversations.json"
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def resolve_episodes_path(path: str | Path) -> Path:
    requested = Path(path)
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.extend([Path.cwd() / requested, REPO_ROOT / requested])
        if requested.name == "conversations.json":
            candidates.append(DEFAULT_EPISODES_PATH)
    for c in dict.fromkeys(candidates):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find episodes file '{path}'. "
        f"Tried: {', '.join(str(c) for c in dict.fromkeys(candidates))}"
    )


def load_episodes(path: str) -> list[list[dict]]:
    resolved = resolve_episodes_path(path)
    logger.info("Loading episodes from %s", resolved)

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list):
        raise ValueError(
            "Unsupported conversations JSON format. "
            "Expected a top-level object with a 'conversations' list."
        )

    raw_episodes = data["conversations"]
    episodes: list[list[dict]] = []
    skipped_episodes = skipped_turns = total_turns = 0

    for ep in raw_episodes:
        if not isinstance(ep, dict):
            skipped_episodes += 1
            continue
        turns = ep.get("turns")
        if not isinstance(turns, list):
            skipped_episodes += 1
            continue

        episode_meta = {
            "episode_id":    ep.get("episode_id", ""),
            "scenario_type": ep.get("scenario_type", "unknown"),
            "scenario_name": ep.get("scenario_name", ""),
            "hazard_type":   ep.get("hazard_type", ""),
            "topic_group":   ep.get("topic_group", ""),
        }
        parsed: list[dict] = []
        for t in turns:
            if not isinstance(t, dict):
                skipped_turns += 1
                continue
            q = t.get("query")
            a = t.get("assistant_answer")
            if q is None or a is None:
                skipped_turns += 1
                continue
            parsed.append({
                **episode_meta,
                "turn_id":            t.get("turn_id"),
                "query":              str(q),
                "ground_truth":       str(a),
                "relevant_chunks":    t.get("relevant_chunks", []),
                "retrieval_required": bool(t.get("retrieval_required", True)),
                "query_type":         t.get("query_type", "unknown"),
            })

        if parsed:
            episodes.append(parsed)
            total_turns += len(parsed)
        else:
            skipped_episodes += 1

    if skipped_episodes:
        logger.warning("Skipped %d malformed episodes", skipped_episodes)
    if skipped_turns:
        logger.warning("Skipped %d malformed turns", skipped_turns)
    logger.info("Loaded %d episodes (%d turns)", len(episodes), total_turns)
    return episodes


def split_episodes(
    episodes: list[list[dict]],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    # test gets the remainder: 1 - train_ratio - val_ratio = 0.15
    seed: int = 42,
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
    """
    Stratified 3-way split: train / val / test.
 
    Stratified by scenario_type so each split has a representative
    distribution of scenario types (A, B, C, D, E, K …).
 
    Returns (train_episodes, val_episodes, test_episodes).
 
    For 3,500 episodes with default ratios:
        train : ~2,450 episodes (~17,150 turns)
        val   :   ~525 episodes (~3,675 turns)   — warm-start early stop + EvaluationCallback
        test  :   ~525 episodes (~3,675 turns)   — final reported metrics ONLY
    """
    import random
    from collections import defaultdict
 
    assert train_ratio + val_ratio < 1.0, (
        f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}"
    )
 
    random.seed(seed)
 
    # ── Group by scenario_type for stratification ──────────────────────────
    buckets: dict[str, list] = defaultdict(list)
    for ep in episodes:
        # scenario_type is on the first turn (propagated from episode meta)
        scenario = ep[0].get("scenario_type", "unknown") if ep else "unknown"
        buckets[scenario].append(ep)
 
    train_eps: list = []
    val_eps:   list = []
    test_eps:  list = []
 
    for scenario, bucket in buckets.items():
        shuffled = bucket.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
 
        n_train = max(1, int(n * train_ratio))
        n_val   = max(1, int(n * val_ratio))
        # test gets whatever is left — never less than 1 if bucket has >= 3
        n_test  = max(0, n - n_train - n_val)
 
        train_eps.extend(shuffled[:n_train])
        val_eps.extend(shuffled[n_train : n_train + n_val])
        test_eps.extend(shuffled[n_train + n_val :])
 
    # Shuffle within each split so episodes aren't grouped by scenario
    random.shuffle(train_eps)
    random.shuffle(val_eps)
    random.shuffle(test_eps)
 
    total = len(train_eps) + len(val_eps) + len(test_eps)
    logger.info(
        "Split: train=%d (%.0f%%)  val=%d (%.0f%%)  test=%d (%.0f%%)",
        len(train_eps), len(train_eps) / total * 100,
        len(val_eps),   len(val_eps)   / total * 100,
        len(test_eps),  len(test_eps)  / total * 100,
    )
 
    _print_split_distribution(train_eps, val_eps, test_eps)
    return train_eps, val_eps, test_eps
 
 
def _print_split_distribution(train_eps, val_eps, test_eps):
    """Print a table showing scenario_type counts per split."""
    from collections import Counter
    def _counts(eps):
        c = Counter(ep[0].get("scenario_type", "unknown") for ep in eps if ep)
        return c
 
    all_scenarios = sorted(set(
        list(_counts(train_eps).keys())
        + list(_counts(val_eps).keys())
        + list(_counts(test_eps).keys())
    ))
 
    tc, vc, sc = _counts(train_eps), _counts(val_eps), _counts(test_eps)
    print("\n  Split distribution by scenario_type:")
    print(f"  {'Scenario':<14} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("  " + "─" * 44)
    for s in all_scenarios:
        t, v, te = tc.get(s, 0), vc.get(s, 0), sc.get(s, 0)
        print(f"  {s:<14} {t:>8} {v:>8} {te:>8} {t+v+te:>8}")
    print("  " + "─" * 44)
    print(
        f"  {'TOTAL':<14} {len(train_eps):>8} {len(val_eps):>8} "
        f"{len(test_eps):>8} {len(train_eps)+len(val_eps)+len(test_eps):>8}"
    )
    print()

# ──────────────────────────────────────────────────────────────────────────────
# RetrievalStatsCallback
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalStatsCallback(BaseCallback):
    """Lightweight per-step stats logged every 500 env steps."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._actions: list[int]   = []
        self._rewards: list[float] = []
        # Accumulated for plotting
        self.rollout_metrics: list[dict] = []

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])

        for act, info in zip(actions, infos):
            self._actions.append(int(act))
            if "score" in info:
                self._rewards.append(float(info["score"]))

        if self.n_calls % 500 == 0 and self._actions:
            ret_rate   = sum(self._actions) / len(self._actions)
            mean_score = float(np.mean(self._rewards)) if self._rewards else float("nan")
            self.logger.record("train/retrieval_rate",   ret_rate)
            self.logger.record("train/mean_judge_score", mean_score)
            _ts = time.strftime("%H:%M:%S")
            print(
                f"  [{_ts}]  step {self.n_calls:>6}  │  "
                f"retrieval {ret_rate * 100:.1f}%  │  "
                f"judge score {mean_score:.2f}"
            )
            self.rollout_metrics.append({
                "event":            "ROLLOUT_SUMMARY",
                "step":             self.n_calls,
                "retrieval_rate":   ret_rate,
                "mean_judge_score": mean_score,
            })
            trace_event(
                "ROLLOUT_SUMMARY",
                step=self.n_calls,
                retrieval_rate_pct=ret_rate * 100,
                mean_judge_score=mean_score,
            )
            self._actions.clear()
            self._rewards.clear()

        return True


# ──────────────────────────────────────────────────────────────────────────────
# Stage A — Supervised Warm Start
# ──────────────────────────────────────────────────────────────────────────────

def _build_warmstart_dataset(
    episodes: list[list[dict]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode every turn into (observations, labels) for warm-start training.

    NOTE: Uses summary="" for all turns (distribution shift vs. inference where
    summary is populated). Acceptable for warm-start — the turn/prev_action
    context still gives the encoder enough signal for classification.
    """
    obs_list:   list[np.ndarray] = []
    label_list: list[int]        = []

    for episode in episodes:
        prev_action = -1
        for turn_idx, turn in enumerate(episode):
            obs = encode_state(
                summary="",
                query=turn["query"],
                turn_number=turn_idx + 1,
                prev_action=prev_action,
            )
            label = int(turn["retrieval_required"])
            obs_list.append(obs)
            label_list.append(label)
            prev_action = label

    obs_array   = np.stack(obs_list).astype(np.float32)
    label_array = np.array(label_list, dtype=np.int64)
    return obs_array, label_array


def _compute_warmstart_metrics(
    all_preds: list[int],
    all_labels: list[int],
) -> dict[str, float]:
    """
    Compute classification metrics:
      accuracy, precision (retrieve), recall (retrieve), F1 (retrieve),
      unsafe_skip_rate  (FN / total — most dangerous failure),
      wasteful_retrieve_rate (FP / total — efficiency waste).
    """
    tp = fp = fn = tn = 0
    for pred, label in zip(all_preds, all_labels):
        if pred == 1 and label == 1:   tp += 1
        elif pred == 1 and label == 0: fp += 1
        elif pred == 0 and label == 1: fn += 1
        else:                          tn += 1

    total     = len(all_preds)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "unsafe_skip_rate":       fn / total if total > 0 else 0.0,
        "wasteful_retrieve_rate": fp / total if total > 0 else 0.0,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def warm_start_train(
    train_episodes: list[list[dict]],
    val_episodes:   list[list[dict]],
    save_path:      str   = "policy_checkpoints",
    lr:             float = 1e-3,
    max_epochs:     int   = 30,
    batch_size:     int   = 64,
    patience:       int   = 5,
    device:         str   = "cpu",
    use_cache:      bool  = True,
) -> str:
    """Stage A — Supervised Warm Start. Returns path to saved .zip checkpoint."""
    print("\n" + "═" * 64)
    print("  Stage A — Supervised Warm Start")
    print("─" * 64)

    # ── Encode (or load from cache) ────────────────────────────────────────
    if use_cache:
        try:
            from rl_core.embed_cache import build_warmstart_dataset_cached
            print(f"  Encoding train turns … (cache enabled)")
            train_obs, train_labels = build_warmstart_dataset_cached(
                train_episodes, split="train"
            )
            print(f"  Encoding val turns   … (cache enabled)")
            val_obs, val_labels = build_warmstart_dataset_cached(
                val_episodes, split="val"
            )
        except Exception as exc:
            logger.warning("Cache failed (%s) — falling back to direct encode", exc)
            use_cache = False

    if not use_cache:
        print(f"  Encoding {len(train_episodes)} train episodes …")
        train_obs, train_labels = _build_warmstart_dataset(train_episodes)
        print(f"  Encoding {len(val_episodes)} val episodes …")
        val_obs, val_labels = _build_warmstart_dataset(val_episodes)

    n_train  = len(train_labels)
    n_val    = len(val_labels)
    ret_rate = train_labels.mean() * 100
    print(f"  Train turns   : {n_train:,}  (retrieve {ret_rate:.1f}%)")
    print(f"  Val turns     : {n_val:,}")
    print(f"  LR={lr}  batch_size={batch_size}  max_epochs={max_epochs}  patience={patience}")
    print("═" * 64 + "\n")

    # ── Build policy object via throw-away PPO ─────────────────────────────
    _dummy_env = VecMonitor(
        DummyVecEnv([lambda: RAGDecisionEnv(train_episodes[:1], mode="train")])
    )
    _dummy_model = PPO(
        policy=TransformerHeadPolicy,
        env=_dummy_env,
        learning_rate=1e-3,
        n_steps=64,
        batch_size=min(batch_size, 64),  # SB3 requires n_steps >= batch_size
        verbose=0,
    )
    policy = _dummy_model.policy.to(device)

    # ── Freeze value head — only train trunk + action head ────
    # The value head parameters are excluded from the optimizer.
    trainable_params = (
        list(policy.mlp_trunk.parameters())
        + list(policy.action_head.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ── Convert arrays to tensors ──────────────────────────────────────────
    X_train = torch.tensor(train_obs,   dtype=torch.float32, device=device)
    y_train = torch.tensor(train_labels, dtype=torch.long,   device=device)
    X_val   = torch.tensor(val_obs,     dtype=torch.float32, device=device)
    y_val   = torch.tensor(val_labels,  dtype=torch.long,    device=device)

    # ── Training loop with early stopping on val F1 ───────────────────────
    best_val_f1     = -1.0
    best_epoch      = 0
    epochs_no_improve = 0
    best_state_dict   = None

    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Acc':>8}  {'Val F1':>7}  "
          f"{'Unsafe Skip':>12}  {'Wasteful Ret':>13}")
    print("  " + "─" * 62)

    for epoch in range(1, max_epochs + 1):
        # ── Mini-batch SGD over training set ──────────────────────────────
        policy.train()
        perm       = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            idx     = perm[start : start + batch_size]
            obs_b   = X_train[idx]
            label_b = y_train[idx]

            # Forward: obs → features → trunk → action_head → logits
            with torch.no_grad():
                # FrozenTransformerExtractor is an identity pass
                features = policy.extract_features(obs_b)
            trunk_out = policy.mlp_trunk(features)
            logits    = policy.action_head(trunk_out)

            loss = criterion(logits, label_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ────────────────────────────────────────────────────
        policy.eval()
        with torch.no_grad():
            val_feats  = policy.extract_features(X_val)
            val_trunk  = policy.mlp_trunk(val_feats)
            val_logits = policy.action_head(val_trunk)
            val_preds  = val_logits.argmax(dim=-1).cpu().numpy().tolist()

        m = _compute_warmstart_metrics(val_preds, y_val.cpu().numpy().tolist())
        print(
            f"  {epoch:>5}  {avg_loss:>11.4f}  {m['accuracy']:>8.3f}  "
            f"{m['f1']:>7.3f}  {m['unsafe_skip_rate']:>12.3f}  "
            f"{m['wasteful_retrieve_rate']:>13.3f}"
        )
        trace_event(
            "WARMSTART_EPOCH", epoch=epoch, train_loss=avg_loss,
            val_accuracy=m["accuracy"], val_f1=m["f1"],
            unsafe_skip_rate=m["unsafe_skip_rate"],
            wasteful_retrieve_rate=m["wasteful_retrieve_rate"],
        )

        if m["f1"] > best_val_f1 + 1e-4:
            best_val_f1 = m["f1"]
            best_epoch  = epoch
            epochs_no_improve = 0
            best_state_dict = {
                "mlp_trunk":   {k: v.clone() for k, v in policy.mlp_trunk.state_dict().items()},
                "action_head": {k: v.clone() for k, v in policy.action_head.state_dict().items()},
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best F1={best_val_f1:.3f} at epoch {best_epoch})")
                break

    # ── FIX: Guard against empty dataset / 0 epochs ───────────────────────
    if best_state_dict is None:
        raise RuntimeError(
            "Warm-start produced no valid checkpoint. "
            f"Ran {epoch} epoch(s), n_train={n_train}. "
            "Check max_epochs > 0 and dataset is non-empty."
        )

    policy.mlp_trunk.load_state_dict(best_state_dict["mlp_trunk"])
    policy.action_head.load_state_dict(best_state_dict["action_head"])

    # ── Final validation metrics ───────────────────────────────────────────
    policy.eval()
    with torch.no_grad():
        val_feats  = policy.extract_features(X_val)
        val_trunk  = policy.mlp_trunk(val_feats)
        val_logits = policy.action_head(val_trunk)
        final_preds = val_logits.argmax(dim=-1).cpu().numpy().tolist()

    final_metrics = _compute_warmstart_metrics(final_preds, y_val.cpu().numpy().tolist())

    # ── Print summary table ────────────────────────────────────────────────
    _hr = "─" * 62
    print(f"\n  {_hr}")
    print(f"  Warm-Start Final Metrics  (best epoch={best_epoch})")
    print(f"  {_hr}")
    print(f"  Accuracy           : {fm['accuracy']:.4f}")
    print(f"  Precision(retrieve): {fm['precision']:.4f}")
    print(f"  Recall(retrieve)   : {fm['recall']:.4f}")
    print(f"  F1(retrieve)       : {fm['f1']:.4f}")
    print(f"  Unsafe Skip Rate   : {fm['unsafe_skip_rate']:.4f}  (FN={fm['FN']})")
    print(f"  Wasteful Ret Rate  : {fm['wasteful_retrieve_rate']:.4f}  (FP={fm['FP']})")
    print(f"  {_hr}\n")

    trace_event(
        "WARMSTART_COMPLETE", best_epoch=best_epoch, best_val_f1=best_val_f1,
        final_accuracy=fm["accuracy"], final_f1=fm["f1"],
        final_unsafe_skip_rate=fm["unsafe_skip_rate"],
        final_wasteful_retrieve_rate=fm["wasteful_retrieve_rate"],
    )

    # ── Save checkpoint via PPO.save() so Stage B can PPO.load() it ───────
    # Update the model's policy with our best weights, then save.
    _dummy_model.policy.mlp_trunk.load_state_dict(best_state_dict["mlp_trunk"])
    _dummy_model.policy.action_head.load_state_dict(best_state_dict["action_head"])

    Path(save_path).mkdir(parents=True, exist_ok=True)
    warmstart_path = str(Path(save_path) / "policy_warmstart")
    _dummy_model.save(warmstart_path)
    print(f"  Warm-start checkpoint saved → {warmstart_path}.zip\n")
    trace_event("WARMSTART_SAVED", path=f"{warmstart_path}.zip")
    return f"{warmstart_path}.zip"


# ──────────────────────────────────────────────────────────────────────────────
# Stage B — PPO Fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def train(
    train_episodes:    list[list[dict]],
    val_episodes:      list[list[dict]],
    total_timesteps:   int  = 3_000,
    save_path:         str  = "policy_checkpoints",
    tensorboard_log:   str  = "./tb_logs",
    trace_log:         str  = "./training_logs/training_trace.log",
    warmstart_weights: str | None = None,
    generate_plots:    bool = True,
    plots_dir:         str  = "plots",
    eval_episodes:     int  = 50,
):
    """
    Stage B — PPO Fine-tuning.

    eval_episodes : number of val episodes to use for EvaluationCallback and
                    post-training evaluate(). Default 50 keeps each pass ~20-30 min.
                    Set to 0 or None to use all val episodes (very slow).
    """
    trace_log_path = configure_training_trace(trace_log)

    # Cap val episodes for evaluation speed
    eval_val = val_episodes
    if eval_episodes and len(val_episodes) > eval_episodes:
        random.seed(42)
        eval_val = random.sample(val_episodes, eval_episodes)
        logger.info(
            "Evaluation capped at %d/%d val episodes for speed",
            eval_episodes, len(val_episodes),
        )

    warm_tag = f"warm-start from {warmstart_weights}" if warmstart_weights else "random init"
    print("\n" + "═" * 64)
    print("  Stage B — PPO Fine-tuning")
    print("─" * 64)
    print(f"  Timesteps     : {total_timesteps:,}")
    print(f"  Train eps     : {len(train_episodes):,}   |   Val eps (eval): {len(eval_val)}")
    print(f"  Actor init    : {warm_tag}")
    print(f"  Save path     : {save_path}")
    print(f"  Trace log     : {trace_log_path}")
    print("═" * 64 + "\n")

    trace_event(
        "TRAINING_START",
        total_timesteps=total_timesteps,
        train_episodes=len(train_episodes),
        val_episodes=len(eval_val),
        warmstart=warmstart_weights or "none",
    )

    logger.info("Building training environment …")
    train_env = VecMonitor(
        DummyVecEnv([lambda: RAGDecisionEnv(train_episodes, mode="train")])
    )

    model = PPO(
        policy=TransformerHeadPolicy,
        env=train_env,
        learning_rate=1e-3,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=0,
    )

    # ── Load warm-start weights into the PPO policy (actor only) ──────────
    # load those weights → keep frozen BERT frozen → initialize PPO
    #           actor from pretrained classifier → value head learned during PPO
    if warmstart_weights is not None:
        wp = Path(warmstart_weights)
        if wp.exists():
            logger.info("Loading warm-start weights from %s", warmstart_weights)
            ws = PPO.load(warmstart_weights)
            model.policy.mlp_trunk.load_state_dict(ws.policy.mlp_trunk.state_dict())
            model.policy.action_head.load_state_dict(ws.policy.action_head.state_dict())
            print(f"  ✓ Actor weights loaded from warm-start checkpoint.\n")
            del ws
        else:
            logger.warning("Warm-start file %s not found — random init.", warmstart_weights)

    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    # Eval every ~10% of training, minimum every 1 update
    updates_per_eval = max(1, total_timesteps // (256 * 10))

    print(f"  PPO model ready  |  trainable params: {trainable:,}")
    print(f"  Eval every {updates_per_eval} update(s)  "
          f"(~{max(1, total_timesteps // (updates_per_eval * 256))} checkpoints)\n")

    eval_callback  = EvaluationCallback(
        val_episodes=eval_val,
        eval_every_n_updates=updates_per_eval,
        save_path=save_path,
        verbose=1,
    )
    stats_callback = RetrievalStatsCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, stats_callback],
        tb_log_name="rag_retrieval_policy",
        progress_bar=True,
    )

    final_path = f"{save_path}/policy_model_final"
    model.save(final_path)
    print(f"\n  Model saved → {final_path}.zip")
    trace_event("TRAINING_END", final_model_path=f"{final_path}.zip")

    # ── Post-training evaluation ───────────────────────────────────────────
    print("\n" + "═" * 80)
    print(f"  POST-TRAINING EVALUATION  ({len(eval_val)} episodes × 4 strategies)")
    print("═" * 80 + "\n")
    t0 = time.time()
    results = evaluate(model, eval_val, save_path=save_path)
    print(f"\n  Evaluation complete ({time.time() - t0:.1f}s)")

    # ── Generate plots ─────────────────────────────────────────────────────
    if generate_plots:
        print("\n  Generating plots …")
        try:
            from rl_core.plotting import generate_all_plots
            training_metrics = (
                stats_callback.rollout_metrics
                + eval_callback.checkpoint_metrics
            )
            plot_dir = generate_all_plots(
                eval_results=results,
                training_metrics=training_metrics,
                save_dir=plots_dir,
            )
            print(f"  Plots saved → {plot_dir}\n")
        except ImportError as exc:
            logger.warning("Plotting skipped (missing dep): %s", exc)
            print("  → Install with: pip install matplotlib seaborn")
        except Exception as exc:
            logger.warning("Plotting failed: %s", exc, exc_info=True)

    return model, results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train retrieval-decision policy  (warm-start + PPO)"
    )
    p.add_argument("--episodes",
        default="dataset_generator/dialog_dataset/conversations.json")
    p.add_argument("--timesteps", type=int, default=3_000,
        help="PPO timesteps. 3000≈2-4hrs, 500≈30-40min smoke test (default: 3000)")
    p.add_argument("--save-path", default="policy_checkpoints")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--policy-path", default=CONFIG.policy_model_path)
    p.add_argument("--tb-log", default="./tb_logs")
    p.add_argument("--trace-log", default="./training_logs/training_trace.log")

    # Warm start
    p.add_argument("--no-warm-start", dest="warm_start",
        action="store_false", default=True)
    p.add_argument("--warm-start-epochs", type=int, default=30)
    p.add_argument("--warm-start-lr", type=float, default=1e-3)
    p.add_argument("--warm-start-patience", type=int, default=5)
    p.add_argument("--warm-start-batch-size", type=int, default=256,
        help="Mini-batch size for warm-start (default: 256 — good for 16k+ turns)")

    # Scale controls
    p.add_argument("--eval-episodes", type=int, default=50,
        help="Number of val episodes for each eval pass. "
             "50 ≈ 20-30min. 0 = use all (very slow). (default: 50)")
    p.add_argument("--no-cache", dest="use_cache",
        action="store_false", default=True,
        help="Disable warm-start embedding cache (always re-encode)")

    # Plotting
    p.add_argument("--no-plots", dest="generate_plots",
        action="store_false", default=True)
    p.add_argument("--plots-dir", default="plots")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
 
    all_episodes = load_episodes(args.episodes)
    train_eps, val_eps, test_eps = split_episodes(all_episodes)
 
    if args.eval_only:
        logger.info("Eval-only: loading from %s", args.policy_path)
        configure_training_trace(args.trace_log)
        model = PPO.load(args.policy_path)
 
        # Eval-only always uses TEST set
        eval_set = test_eps
        if args.eval_episodes and len(eval_set) > args.eval_episodes:
            random.seed(42)
            eval_set = random.sample(eval_set, args.eval_episodes)
 
        print(f"  Evaluating on TEST set ({len(eval_set)} episodes)")
        results = evaluate(model, eval_set, save_path=args.save_path)
 
        if args.generate_plots:
            try:
                from rl_core.plotting import generate_all_plots
                plot_dir = generate_all_plots(
                    eval_results=results,
                    training_metrics=None,
                    save_dir=args.plots_dir,
                )
                print(f"  Plots saved -> {plot_dir}")
            except Exception as exc:
                logger.warning("Plotting failed: %s", exc)
 
    else:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        configure_training_trace(args.trace_log)
 
        warmstart_weights = None
 
        if args.warm_start:
            # Warm-start uses TRAIN + VAL (val for early stopping only)
            warmstart_weights = warm_start_train(
                train_episodes      = train_eps,
                val_episodes        = val_eps,
                save_path           = args.save_path,
                lr                  = args.warm_start_lr,
                max_epochs          = args.warm_start_epochs,
                batch_size          = args.warm_start_batch_size,
                patience            = args.warm_start_patience,
                use_cache           = args.use_cache,
            )
        else:
            print("  [--no-warm-start]  Skipping Stage A.")
            trace_event("WARMSTART_SKIPPED", reason="--no-warm-start flag set")
 
        # PPO uses TRAIN env + VAL for EvaluationCallback checkpoints
        model, _ = train(
            train_episodes    = train_eps,
            val_episodes      = val_eps,       # <-- val: used during training
            total_timesteps   = args.timesteps,
            save_path         = args.save_path,
            tensorboard_log   = args.tb_log,
            trace_log         = args.trace_log,
            warmstart_weights = warmstart_weights,
            generate_plots    = False,          # <-- no plots yet; test set not run
            plots_dir         = args.plots_dir,
            eval_episodes     = args.eval_episodes,
        )
 
        # Final evaluation ALWAYS on TEST set — never seen during training
        eval_set = test_eps
        if args.eval_episodes and len(eval_set) > args.eval_episodes:
            random.seed(42)
            eval_set = random.sample(eval_set, args.eval_episodes)
 
        print("\\n" + "=" * 80)
        print(f"  FINAL EVALUATION ON TEST SET  ({len(eval_set)} episodes)")
        print("=" * 80 + "\\n")
        results = evaluate(model, eval_set, save_path=args.save_path)
 
        if args.generate_plots:
            try:
                from rl_core.plotting import generate_all_plots
                # Grab training metrics from the train() internals via checkpoint files
                # (simplest approach: plots from eval results only at this stage)
                plot_dir = generate_all_plots(
                    eval_results=results,
                    training_metrics=None,
                    save_dir=args.plots_dir,
                )
                print(f"  Plots saved -> {plot_dir}")
            except Exception as exc:
                logger.warning("Plotting failed: %s", exc)
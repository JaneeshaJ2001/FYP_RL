"""
rl_train.py
───────────
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
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from collections import defaultdict
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

# Suppress noisy third-party loggers
for _noisy_logger in ("stable_baselines3", "httpx", "httpcore", "langfuse", "openai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODES_PATH = (
    REPO_ROOT / "dataset_generator" / "dialog_dataset" / "conversations.json"
)

# Any scenario group with fewer episodes than this cannot produce meaningful
# val and test cells — those episodes go entirely into train.
_MIN_EPISODES_TO_SPLIT = 30


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def resolve_episodes_path(path: str | Path) -> Path:
    requested_path = Path(path)
    candidates: list[Path] = []

    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.extend([
            Path.cwd() / requested_path,
            REPO_ROOT / requested_path,
        ])
        if requested_path.name == "conversations.json":
            candidates.append(DEFAULT_EPISODES_PATH)

    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            return candidate

    searched_paths = ", ".join(str(c) for c in dict.fromkeys(candidates))
    raise FileNotFoundError(
        f"Could not find episodes file '{path}'. Tried: {searched_paths}"
    )


def load_episodes(path: str) -> list[list[dict]]:
    resolved_path = resolve_episodes_path(path)
    logger.info("Loading episodes from %s", resolved_path)

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("conversations"), list):
        raise ValueError(
            "Unsupported conversations JSON format. Expected a top-level object "
            "with a 'conversations' list."
        )

    raw_episodes = data["conversations"]
    episodes: list[list[dict]] = []
    skipped_episodes = 0
    skipped_turns = 0
    total_turns = 0

    for ep in raw_episodes:
        if not isinstance(ep, dict):
            skipped_episodes += 1
            logger.warning("Skipping malformed episode that is not an object")
            continue

        turns = ep.get("turns")
        if not isinstance(turns, list):
            logger.warning("Skipping malformed episode with non-list turns")
            skipped_episodes += 1
            continue

        episode_meta = {
            "episode_id":    ep.get("episode_id", ""),
            "scenario_type": ep.get("scenario_type", "unknown"),
            "scenario_name": ep.get("scenario_name", ""),
            "hazard_type":   ep.get("hazard_type", ""),
            "topic_group":   ep.get("topic_group", ""),
        }

        parsed_turns: list[dict] = []
        for t in turns:
            if not isinstance(t, dict):
                skipped_turns += 1
                continue

            query            = t.get("query")
            assistant_answer = t.get("assistant_answer")

            if query is None or assistant_answer is None:
                skipped_turns += 1
                continue

            parsed_turn = {
                **episode_meta,
                "turn_id":            t.get("turn_id"),
                "query":              str(query),
                "ground_truth":       str(assistant_answer),
                "relevant_chunks":    t.get("relevant_chunks", []),
                "retrieval_required": bool(t.get("retrieval_required", True)),
                "query_type":         t.get("query_type", "unknown"),
            }
            parsed_turns.append(parsed_turn)

        if parsed_turns:
            episodes.append(parsed_turns)
            total_turns += len(parsed_turns)
        else:
            skipped_episodes += 1

    if skipped_episodes:
        logger.warning("Skipped %d malformed episodes", skipped_episodes)
    if skipped_turns:
        logger.warning("Skipped %d malformed turns", skipped_turns)
    logger.info("Loaded %d episodes (%d turns)", len(episodes), total_turns)
    return episodes


# ──────────────────────────────────────────────────────────────────────────────
# split_episodes  — three-way, stratified by scenario_type
# ──────────────────────────────────────────────────────────────────────────────

def split_episodes(
    episodes: list[list[dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    stratify_key: str = "scenario_type",
    min_split_size: int = _MIN_EPISODES_TO_SPLIT,
) -> tuple[list[list[dict]], list[list[dict]], list[list[dict]]]:
    """
    Split episodes into train / val / test at the EPISODE level.

    Stratification guarantees all scenario_type classes appear
    proportionally in every split.
    Groups below min_split_size go entirely into train.

    Parameters
    ----------
    episodes       : output of load_episodes() — list[list[dict]]
                     outer list = episodes, inner list = turns
    train_ratio    : 0.70 → 70 % of episodes to train
    val_ratio      : 0.15 → 15 % to val  (used by EvaluationCallback
                     and warm_start validation only — NOT the final report)
    seed           : random seed
    stratify_key   : turn-level key propagated from episode metadata.
                     "scenario_type" has values A-N in this dataset.
    min_split_size : groups smaller than this go entirely into train.
                     At 70/15/15 this threshold of 30 ensures every split
                     cell has at least 4 val and 4 test episodes.

    Returns
    -------
    (train_episodes, val_episodes, test_episodes)
    test_episodes is held out and used ONLY for the final evaluate() call.
    """
    random.seed(seed)

    # ── Group by stratification key (episode level) ────────────────────────
    groups: dict[str, list[list[dict]]] = defaultdict(list)
    for ep in episodes:
        key_val = ep[0].get(stratify_key, "__unknown__") if ep else "__unknown__"
        groups[key_val].append(ep)

    train: list[list[dict]] = []
    val:   list[list[dict]] = []
    test:  list[list[dict]] = []
    thin_groups: list[str]  = []

    for group_key in sorted(groups.keys()):   # sorted → reproducible
        group_eps = groups[group_key]
        random.shuffle(group_eps)
        n = len(group_eps)

        if n < min_split_size:
            # Too few episodes for reliable val/test metrics — train only.
            train.extend(group_eps)
            thin_groups.append(group_key)
            continue

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val          # remainder to test

        if n_train < 1 or n_val < 1 or n_test < 1:
            logger.warning(
                "Group '%s' (%d eps) produced an empty split slice — "
                "placing all in train.", group_key, n,
            )
            train.extend(group_eps)
            thin_groups.append(group_key)
            continue

        train.extend(group_eps[:n_train])
        val.extend(group_eps[n_train : n_train + n_val])
        test.extend(group_eps[n_train + n_val :])

    # Shuffle so episodes from the same group are not contiguous
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    if thin_groups:
        logger.warning(
            "Thin groups (n < %d) placed entirely in train: %s. "
            "Exclude from per-scenario val/test breakdown in reports.",
            min_split_size, sorted(thin_groups),
        )

    logger.info(
        "Split (stratified by %s, seed=%d): train=%d  val=%d  test=%d",
        stratify_key, seed, len(train), len(val), len(test),
    )

    _print_split_table(train, val, test, stratify_key, thin_groups)
    return train, val, test


def _print_split_table(
    train: list[list[dict]],
    val:   list[list[dict]],
    test:  list[list[dict]],
    key:   str,
    thin_groups: list[str],
) -> None:
    """Print a verification table so imbalances are visible before training starts."""
    from collections import Counter

    def dist(split: list[list[dict]]) -> Counter:
        c: Counter = Counter()
        for ep in split:
            v = ep[0].get(key, "?") if ep else "?"
            c[v] += 1
        return c

    tr, vl, te = dist(train), dist(val), dist(test)
    all_keys = sorted(set(tr) | set(vl) | set(te))
    n_tr, n_vl, n_te = len(train), len(val), len(test)
    n_total = n_tr + n_vl + n_te
    avg_turns = 5.74

    W = 72
    print(f"\n  {'═' * W}")
    print(f"  Split verification  (stratified by {key})")
    print(f"  {int(n_tr/n_total*100)}% train / "
          f"{int(n_vl/n_total*100)}% val / "
          f"{int(n_te/n_total*100)}% test")
    print(f"  {'─' * W}")
    print(f"  {'Scen':<6} {'Total':>7} {'Train':>8} {'Val':>8} {'Test':>8}   Note")
    print(f"  {'─' * W}")
    for k in all_keys:
        total_k = tr[k] + vl[k] + te[k]
        note = "thin → train only" if k in thin_groups else ""
        print(f"  {k:<6} {total_k:>7} {tr[k]:>8} {vl[k]:>8} {te[k]:>8}   {note}")
    print(f"  {'─' * W}")
    print(f"  {'TOTAL':<6} {n_total:>7,} {n_tr:>8,} {n_vl:>8,} {n_te:>8,}")
    print(f"  {'TURNS':<6} {int(n_total*avg_turns):>7,} "
          f"{int(n_tr*avg_turns):>8,} {int(n_vl*avg_turns):>8,} "
          f"{int(n_te*avg_turns):>8,}   (est. @ {avg_turns} turns/ep)")
    if thin_groups:
        print(f"  ⚠  Thin groups (train only): {', '.join(sorted(thin_groups))}")
    print(f"  {'═' * W}\n")


# ──────────────────────────────────────────────────────────────────────────────
# RetrievalStatsCallback
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalStatsCallback(BaseCallback):
    """Lightweight per-step stats logged every 500 env steps."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._actions: list[int]   = []
        self._rewards: list[float] = []

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])

        for act, info in zip(actions, infos):
            self._actions.append(int(act))
            if "score" in info:
                self._rewards.append(float(info["score"]))

        if self.n_calls % 500 == 0 and self._actions:
            retrieve_rate = sum(self._actions) / len(self._actions)
            mean_reward   = float(np.mean(self._rewards)) if self._rewards else float("nan")
            self.logger.record("train/retrieval_rate",   retrieve_rate)
            self.logger.record("train/mean_judge_score", mean_reward)
            _ts = time.strftime("%H:%M:%S")
            print(
                f"  [{_ts}]  step {self.n_calls:>6}  │  "
                f"retrieval {retrieve_rate * 100:.1f}%  │  "
                f"judge score {mean_reward:.2f}"
            )
            trace_event(
                "ROLLOUT_SUMMARY",
                step=self.n_calls,
                retrieval_rate_pct=retrieve_rate * 100,
                mean_judge_score=mean_reward,
            )
            self._actions.clear()
            self._rewards.clear()

        return True


# ──────────────────────────────────────────────────────────────────────────────
# === SUPERVISED WARM START STAGE ===
# ──────────────────────────────────────────────────────────────────────────────

def _build_warmstart_dataset(
    episodes: list[list[dict]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode every turn in `episodes` into (observations, labels).

    Observation : encode_state(summary="", query=query, turn_number=turn_idx+1,
                               prev_action=prev_action)
                  We use an empty summary here because during warm start we
                  don't run the full graph — we just need a consistent, fast
                  observation that matches the RL env at step 0 of each turn.

    Label       : int(retrieval_required)  — 0 = skip, 1 = retrieve.

    Returns
    -------
    obs_array   : float32 ndarray, shape (N, encoder_dim)
    label_array : int64 ndarray,   shape (N,)
    """
    obs_list:   list[np.ndarray] = []
    label_list: list[int]        = []

    for episode in episodes:
        prev_action = -1   # -1 sentinel: first turn
        for turn_idx, turn in enumerate(episode):
            obs = encode_state(
                summary="",               # empty summary during supervised encoding
                query=turn["query"],
                turn_number=turn_idx + 1,
                prev_action=prev_action,
            )
            label = int(turn["retrieval_required"])
            obs_list.append(obs)
            label_list.append(label)
            prev_action = label  # use the ground-truth label as prev_action for next turn

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
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    total     = len(all_preds)
    precision = tp / (tp + fp)       if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy              = (tp + tn) / total if total > 0 else 0.0
    unsafe_skip_rate      = fn / total        if total > 0 else 0.0
    wasteful_retrieve_rate = fp / total       if total > 0 else 0.0

    return {
        "accuracy":               accuracy,
        "precision":              precision,
        "recall":                 recall,
        "f1":                     f1,
        "unsafe_skip_rate":       unsafe_skip_rate,
        "wasteful_retrieve_rate": wasteful_retrieve_rate,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def warm_start_train(
    train_episodes: list[list[dict]],
    val_episodes:   list[list[dict]],
    save_path:      str  = "policy_checkpoints",
    lr:             float = 1e-3,
    max_epochs:     int   = 30,
    batch_size:     int   = 64,
    patience:       int   = 5,
    device:         str   = "cpu",
) -> str:
    """
    Stage A — Supervised Warm Start

    Treats TransformerHeadPolicy as a binary classifier:
        observation → frozen encoder → MLP trunk → action head → logits
        loss = CrossEntropyLoss(logits, retrieval_required_label)

    Only trains:  mlp_trunk  +  action_head
    Value head:   left random  (PPO will learn it in Stage B)

    Parameters
    ----------
    train_episodes, val_episodes : episode lists from load_episodes()
    save_path   : directory where "policy_warmstart.zip" will be saved
    lr          : AdamW learning rate (default 1e-3 — only the small head)
    max_epochs  : hard cap on training epochs
    batch_size  : mini-batch size for gradient steps
    patience    : early-stopping patience on val F1 (# epochs without improvement)
    device      : "cpu" or "cuda"

    Returns
    -------
    Path to the saved warm-start checkpoint (str), suitable for PPO.load().
    """
    print("\n" + "═" * 64)
    print("  Stage A — Supervised Warm Start")
    print("─" * 64)
    print(f"  Encoding training turns …")

    # ── Build datasets ─────────────────────────────────────────────────────
    train_obs, train_labels = _build_warmstart_dataset(train_episodes)
    val_obs,   val_labels   = _build_warmstart_dataset(val_episodes)

    n_train = len(train_labels)
    n_val   = len(val_labels)
    ret_rate = train_labels.mean() * 100
    print(f"  Train turns   : {n_train}  (retrieve {ret_rate:.1f}%)")
    print(f"  Val turns     : {n_val}")
    print(f"  LR={lr}  batch_size={batch_size}  max_epochs={max_epochs}  patience={patience}")
    print("═" * 64 + "\n")

    # ── Build a throw-away PPO model just to get the policy object ─────────
    # We need a real Gym env to construct PPO; we use a single-episode env.
    _dummy_env = VecMonitor(
        DummyVecEnv([lambda: RAGDecisionEnv(train_episodes[:1], mode="train")])
    )
    _dummy_model = PPO(
        policy=TransformerHeadPolicy,
        env=_dummy_env,
        learning_rate=1e-3,
        n_steps=64,
        batch_size=batch_size,
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
    best_state_dict = None   # save trunk + action_head weights at best epoch

    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Acc':>8}  {'Val F1':>7}  "
          f"{'Unsafe Skip':>12}  {'Wasteful Ret':>13}")
    print("  " + "─" * 62)

    for epoch in range(1, max_epochs + 1):
        # ── Mini-batch SGD over training set ──────────────────────────────
        policy.train()
        perm      = torch.randperm(n_train, device=device)
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
            val_features  = policy.extract_features(X_val)
            val_trunk     = policy.mlp_trunk(val_features)
            val_logits    = policy.action_head(val_trunk)
            val_preds_t   = val_logits.argmax(dim=-1)

        val_preds  = val_preds_t.cpu().numpy().tolist()
        val_labels_list = y_val.cpu().numpy().tolist()
        metrics    = _compute_warmstart_metrics(val_preds, val_labels_list)

        val_f1      = metrics["f1"]
        val_acc     = metrics["accuracy"]
        unsafe_skip = metrics["unsafe_skip_rate"]
        wasteful    = metrics["wasteful_retrieve_rate"]

        print(
            f"  {epoch:>5}  {avg_loss:>11.4f}  {val_acc:>8.3f}  "
            f"{val_f1:>7.3f}  {unsafe_skip:>12.3f}  {wasteful:>13.3f}"
        )

        trace_event(
            "WARMSTART_EPOCH",
            epoch=epoch,
            train_loss=avg_loss,
            val_accuracy=val_acc,
            val_f1=val_f1,
            unsafe_skip_rate=unsafe_skip,
            wasteful_retrieve_rate=wasteful,
        )

        # ── Early stopping ─────────────────────────────────────────────────
        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1    = val_f1
            best_epoch     = epoch
            epochs_no_improve = 0
            # Snapshot trunk + action_head state dicts
            best_state_dict = {
                "mlp_trunk":   {k: v.clone() for k, v in policy.mlp_trunk.state_dict().items()},
                "action_head": {k: v.clone() for k, v in policy.action_head.state_dict().items()},
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val F1={best_val_f1:.3f} at epoch {best_epoch})")
                break

    # ── Restore best weights ───────────────────────────────────────────────
    if best_state_dict is not None:
        policy.mlp_trunk.load_state_dict(best_state_dict["mlp_trunk"])
        policy.action_head.load_state_dict(best_state_dict["action_head"])

    # ── Final validation metrics ───────────────────────────────────────────
    policy.eval()
    with torch.no_grad():
        val_features = policy.extract_features(X_val)
        val_trunk    = policy.mlp_trunk(val_features)
        val_logits   = policy.action_head(val_trunk)
        final_preds  = val_logits.argmax(dim=-1).cpu().numpy().tolist()

    final_metrics = _compute_warmstart_metrics(final_preds, y_val.cpu().numpy().tolist())

    # ── Print summary table ────────────────────────────────────────────────
    _hr = "─" * 62
    print(f"\n  {_hr}")
    print(f"  Warm-Start Final Metrics  (best epoch={best_epoch})")
    print(f"  {_hr}")
    print(f"  Accuracy          : {final_metrics['accuracy']:.4f}")
    print(f"  Precision(retrieve): {final_metrics['precision']:.4f}")
    print(f"  Recall(retrieve)  : {final_metrics['recall']:.4f}")
    print(f"  F1(retrieve)      : {final_metrics['f1']:.4f}")
    print(f"  Unsafe Skip Rate  : {final_metrics['unsafe_skip_rate']:.4f}  "
          f"(FN={final_metrics['FN']})")
    print(f"  Wasteful Ret Rate : {final_metrics['wasteful_retrieve_rate']:.4f}  "
          f"(FP={final_metrics['FP']})")
    print(f"  {_hr}\n")

    trace_event(
        "WARMSTART_COMPLETE",
        best_epoch=best_epoch,
        best_val_f1=best_val_f1,
        final_accuracy=final_metrics["accuracy"],
        final_f1=final_metrics["f1"],
        final_unsafe_skip_rate=final_metrics["unsafe_skip_rate"],
        final_wasteful_retrieve_rate=final_metrics["wasteful_retrieve_rate"],
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
# === STAGE B — PPO FINE-TUNING ===
# ──────────────────────────────────────────────────────────────────────────────

def train(
    train_episodes:    list[list[dict]],
    val_episodes:      list[list[dict]],
    total_timesteps:   int  = 20_000,
    save_path:         str  = "policy_checkpoints",
    tensorboard_log:   str  = "./tb_logs",
    trace_log:         str  = "./training_logs/training_trace.log",
    warmstart_weights: str | None = None,
):
    """
    Stage B — PPO Fine-tuning.

    If `warmstart_weights` is provided (path to a .zip saved by warm_start_train),
    the PPO policy's MLP trunk and action head are initialised from those weights
    before training begins.  The value head is always learned from scratch.

    All other PPO hyperparameters are UNCHANGED from the original.
    """
    trace_log_path = configure_training_trace(trace_log)

    # ── Training banner ───────────────────────────────────────────────────
    warm_tag = f"warm-start from {warmstart_weights}" if warmstart_weights else "random init"
    print("\n" + "═" * 64)
    print("  Stage B — PPO Fine-tuning")
    print("─" * 64)
    print(f"  Timesteps     : {total_timesteps:,}")
    print(f"  Train eps     : {len(train_episodes)}   |   Val eps : {len(val_episodes)}")
    print(f"  Actor init    : {warm_tag}")
    print(f"  Save path     : {save_path}")
    print(f"  Trace log     : {trace_log_path}")
    print("═" * 64 + "\n")

    trace_event(
        "TRAINING_START",
        total_timesteps=total_timesteps,
        train_episodes=len(train_episodes),
        val_episodes=len(val_episodes),
        warmstart=warmstart_weights or "none",
        save_path=save_path,
        tensorboard_log=tensorboard_log,
        trace_log=trace_log_path,
    )

    logger.info("Building training environment...")
    train_env = VecMonitor(
        DummyVecEnv([lambda: RAGDecisionEnv(train_episodes, mode="train")])
    )

    # ── PPO configuration (UNCHANGED from original) ───────────────────────
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
        warmstart_path = Path(warmstart_weights)
        if warmstart_path.exists():
            logger.info("Loading warm-start weights from %s", warmstart_weights)
            ws_model = PPO.load(warmstart_weights)

            # Copy trunk + action_head; leave value_head untouched (random)
            model.policy.mlp_trunk.load_state_dict(
                ws_model.policy.mlp_trunk.state_dict()
            )
            model.policy.action_head.load_state_dict(
                ws_model.policy.action_head.state_dict()
            )
            print(f"  ✓ Actor weights loaded from warm-start checkpoint.\n")
            del ws_model  # free memory
        else:
            logger.warning(
                "Warm-start file %s not found — starting PPO from random init.",
                warmstart_weights,
            )

    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )

    updates_per_eval = max(1, total_timesteps // (256 * 10))

    print(f"  PPO model ready  |  trainable params: {trainable_params:,}")
    print(
        f"  Eval checkpoints : every {updates_per_eval} update(s)  "
        f"(~{total_timesteps // (updates_per_eval * 256)} checkpoints total)\n"
    )

    eval_callback  = EvaluationCallback(
        val_episodes=val_episodes,
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

    return model


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train retrieval-decision policy with (optional) supervised warm start + PPO"
    )
    p.add_argument(
        "--episodes",
        default="dataset_generator/dialog_dataset/conversations.json",
        help="Path to conversations JSON file",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="Total PPO training timesteps (Stage B)",
    )
    p.add_argument(
        "--save-path",
        default="policy_checkpoints",
        help="Directory to save policy checkpoints",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load saved policy and run evaluation only",
    )
    p.add_argument(
        "--policy-path",
        default=CONFIG.policy_model_path,
        help="Path to .zip policy file for eval-only mode",
    )
    p.add_argument(
        "--tb-log",
        default="./tb_logs",
        help="TensorBoard log directory",
    )
    p.add_argument(
        "--trace-log",
        default="./training_logs/training_trace.log",
        help="Path to detailed line-by-line training trace file",
    )

    # === SUPERVISED WARM START FLAGS ===
    p.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        default=True,
        help="Disable Stage A supervised warm start.",
    )
    p.add_argument(
        "--warm-start-epochs",
        type=int,
        default=30,
    )
    p.add_argument(
        "--warm-start-lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--warm-start-patience",
        type=int,
        default=5,
    )
    p.add_argument(
        "--warm-start-batch-size",
        type=int,
        default=64,
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    all_episodes = load_episodes(args.episodes)

    # THREE-WAY SPLIT — stratified by scenario_type at episode level.
    # val_episodes  → used by EvaluationCallback and warm_start validation only.
    # test_episodes → held out; used only for the final evaluate() call below.
    train_eps, val_eps, test_eps = split_episodes(all_episodes)

    if args.eval_only:
        logger.info("Eval-only mode: loading policy from %s", args.policy_path)
        configure_training_trace(args.trace_log)
        model = PPO.load(args.policy_path)
        # Evaluate on held-out test set, not val
        evaluate(model, test_eps, save_path=args.save_path)

    else:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        configure_training_trace(args.trace_log)

        # Stage A: Supervised Warm Start (trains on train_eps, validates on val_eps)
        warmstart_weights: str | None = None
        if args.warm_start:
            warmstart_weights = warm_start_train(
                train_episodes   = train_eps,
                val_episodes     = val_eps,      # val only — test never touched here
                save_path        = args.save_path,
                lr               = args.warm_start_lr,
                max_epochs       = args.warm_start_epochs,
                batch_size       = args.warm_start_batch_size,
                patience         = args.warm_start_patience,
            )
        else:
            print("\n  [--no-warm-start]  Skipping Stage A — PPO starts from random init.\n")
            trace_event("WARMSTART_SKIPPED", reason="--no-warm-start flag set")

        # Stage B: PPO (trains on train_eps, checkpoints evaluated on val_eps)
        model = train(
            train_episodes    = train_eps,
            val_episodes      = val_eps,         # val only — test never touched here
            total_timesteps   = args.timesteps,
            save_path         = args.save_path,
            tensorboard_log   = args.tb_log,
            trace_log         = args.trace_log,
            warmstart_weights = warmstart_weights,
        )

        # Final evaluation on the HELD-OUT test set
        print("\n" + "═" * 80)
        print("  FINAL EVALUATION  —  held-out test set")
        print("  Comparing: Policy (RL) vs Always-Retrieve vs Always-Skip vs Heuristic-Router")
        print("─" * 80)
        print(f"  Evaluating on {len(test_eps)} test episodes...")
        print(f"  Results → {args.save_path}/final_evaluation/")
        print("═" * 80 + "\n")

        t0 = time.time()
        evaluate(model, test_eps, save_path=args.save_path)   # test, not val
        print(f"\n  Evaluation complete ({time.time()-t0:.1f}s)")
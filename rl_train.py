"""
rl_train.py
───────────
Train and evaluate the retrieval-decision policy with PPO.

Usage
─────
  # Full training run:
    python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --timesteps 20000

  # Quick smoke-test:
    python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --timesteps 500

  # Eval only (load saved policy):
    python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from core.config import CONFIG
from rl_core.rl_env import RAGDecisionEnv
from rl_core.rl_policy import TransformerHeadPolicy
from rl_core.training_trace import configure_training_trace, trace_event

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_train")

# Suppress noisy third-party loggers (HTTP clients, LangFuse telemetry, SB3 internals)
for _noisy_logger in ("stable_baselines3", "httpx", "httpcore", "langfuse", "openai"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EPISODES_PATH = REPO_ROOT / "dataset_generator" / "dialog_dataset" / "conversations.json"

# λ for Utility = Q - λC  — reuses the same β from RL reward
UTILITY_LAMBDA = CONFIG.rl_beta


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
    """
    Load the dialog dataset. Preserves ALL new fields from the dataset schema:

    Episode-level (merged into each turn for easy access during eval):
      episode_id, scenario_type, scenario_name, hazard_type, topic_group

    Turn-level:
      turn_id, query, assistant_answer (stored as ground_truth),
      relevant_chunks, retrieval_required, query_type"
    """
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

        # Episode-level metadata — injected into every turn for breakdown analysis
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
                # turn-level fields
                "turn_id":            t.get("turn_id"),
                "query":              str(query),
                "ground_truth":       str(assistant_answer),   # canonical name used by RL code
                "relevant_chunks":    t.get("relevant_chunks", []),
                "retrieval_required": bool(t.get("retrieval_required", True)),  # safe default
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


def split_episodes(
    episodes: list[list[dict]],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[list[dict]], list[list[dict]]]:
    random.seed(seed)
    shuffled = episodes.copy()
    random.shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_ratio))
    train, val = shuffled[:split], shuffled[split:]
    if not val:
        val = [train[-1]]
    logger.info("Train episodes: %d | Val episodes: %d", len(train), len(val))
    return train, val


# ──────────────────────────────────────────────────────────────────────────────
# Routing metrics helper: confusion matrix + TP/FP/FN/TN rates
# ──────────────────────────────────────────────────────────────────────────────

def compute_routing_metrics(
    actions: list[int],
    retrieval_required: list[bool],
) -> dict[str, float]:
    """
      TP = retrieval chosen  AND retrieval required
      FP = retrieval chosen  BUT skip sufficient
      FN = skip chosen       BUT retrieval required
      TN = skip chosen       AND skip sufficient

    Returns: TP, FP, FN, TN, accuracy, precision, recall, f1,
             unsafe_skip_rate (FN/total), wasteful_retrieve_rate (FP/total)
    """
    assert len(actions) == len(retrieval_required), "length mismatch"
    tp = fp = fn = tn = 0
    for act, req in zip(actions, retrieval_required):
        if act == 1 and req:
            tp += 1
        elif act == 1 and not req:
            fp += 1
        elif act == 0 and req:
            fn += 1
        else:
            tn += 1

    total     = len(actions)
    precision = tp / (tp + fp)       if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)       if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total    if total > 0 else 0.0
    unsafe_skip_rate       = fn / total if total > 0 else 0.0  # dangerous errors  (PDF p.9)
    wasteful_retrieve_rate = fp / total if total > 0 else 0.0  # inefficiency      (PDF p.9)

    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy":               accuracy,
        "precision":              precision,
        "recall":                 recall,
        "f1":                     f1,
        "unsafe_skip_rate":       unsafe_skip_rate,
        "wasteful_retrieve_rate": wasteful_retrieve_rate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Heuristic router
# ──────────────────────────────────────────────────────────────────────────────

def heuristic_action(turn: dict, turn_idx: int) -> int:
    """
    Simple heuristic baseline (PDF p.12):
      - Retrieve on first turn (always needs grounding)
      - Retrieve on topic_shift or followup_new (topic changes need new docs)
      - Skip on same-topic follow-ups, clarifications, unanswerable turns
    """
    query_type = turn.get("query_type", "")
    if turn_idx == 0 or query_type in ("first_turn", "topic_shift", "followup_new"):
        return 1
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Single-episode runner  (shared by EvaluationCallback and evaluate())
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    episode: list[dict],
    *,
    app,
    judge,
    forced_action: int | None = None,
    use_policy: bool = False,
    use_heuristic: bool = False,
    model: PPO | None = None,
    collect_samples: bool = False,
) -> dict[str, Any]:
    """
    Execute one episode under a chosen strategy and return per-turn records.

    Each record contains: query, action, retrieval_required, score, tokens,
    latency, response (if collect_samples=True), and all scenario metadata.
    """
    from rl_core.utils import approx_token_count
    from rl_core.state_encoder import encode_state
    from langchain_core.messages import HumanMessage
    from core.observability import build_run_config
    from core.nodes import _format_docs as _fmt

    summary   = ""
    thread_id = f"eval-{random.randint(0, 10_000_000)}"
    run_config = build_run_config(thread_id)
    turn_records: list[dict] = []

    for t_idx, turn in enumerate(episode):
        query              = turn["query"]
        retrieval_required = turn.get("retrieval_required", True)

        # ── Choose action ──────────────────────────────────────────────────
        if use_policy and model is not None:
            # use deterministic action selection
            obs = encode_state(summary, query)
            action_arr, _ = model.predict(obs, deterministic=True)
            action = int(action_arr)
        elif use_heuristic:
            action = heuristic_action(turn, t_idx)
        else:
            action = int(forced_action)  # type: ignore[arg-type]

        # ── Run LangGraph ──────────────────────────────────────────────────
        input_state = {
            "messages":       [HumanMessage(content=query)],
            "query":          query,
            "summary":        summary,
            "retrieved_docs": [],
            "answer":         "",
            "mode":           "baseline",
        }
        run_config["configurable"]["forced_action"] = action

        t_start = time.time()
        final_state: dict = {}
        try:
            for event in app.stream(input_state, config=run_config):
                for _, partial in event.items():
                    if partial:
                        final_state.update(partial)
        except Exception as exc:
            logger.warning("Graph error in eval at turn %d: %s", t_idx, exc)
        latency = time.time() - t_start

        response = final_state.get("answer", "")
        docs     = final_state.get("retrieved_docs", [])
        summary  = final_state.get("summary", summary)
        tokens   = approx_token_count(docs)

        # ── Judge scoring ──────────────────────────────────────────────────
        docs_text = _fmt(docs) if docs else ""
        result = judge.invoke({
            "query":                query,
            "conversation_summary": summary,
            "action_taken":         "retrieve" if action == 1 else "skip",
            "retrieved_docs":       docs_text,
            "response":             response,
        })
        score = float(result.get("score", 5))

        turn_records.append({
            "query":              query,
            "action":             action,
            "retrieval_required": retrieval_required,
            "score":              score,
            "tokens":             tokens,
            "latency":            latency,
            # Only store text when building sample predictions (saves memory)
            "response":           response     if collect_samples else "",
            "judge_reason":       result.get("reason", "") if collect_samples else "",
            # Scenario metadata — needed for per-scenario breakdown (PDF p.8)
            "scenario_type":  turn.get("scenario_type", "unknown"),
            "scenario_name":  turn.get("scenario_name", ""),
            "query_type":     turn.get("query_type", "unknown"),
            "hazard_type":    turn.get("hazard_type", ""),
            "topic_group":    turn.get("topic_group", ""),
            "episode_id":     turn.get("episode_id", ""),
            "turn_id":        turn.get("turn_id"),
        })

    return {"turns": turn_records}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _aggregate_turns(all_turns: list[dict]) -> dict[str, Any]:
    """
    Aggregate a flat list of turn records into summary metrics.

    PDF p.2 metrics: avg_reward, avg_judge_score, retrieval_rate, unsafe_skip_rate,
                     wasteful_retrieve_rate
    PDF p.12: Utility = Q - λC where Q = judge_score/10, C = tokens/1000
    """
    n = len(all_turns)
    if n == 0:
        return {}

    scores    = [t["score"]   for t in all_turns]
    tokens    = [t["tokens"]  for t in all_turns]
    actions   = [t["action"]  for t in all_turns]
    req       = [t["retrieval_required"] for t in all_turns]
    latencies = [t["latency"] for t in all_turns]

    # Reward mirrors rl_env formula: score - β * tokens
    rewards   = [s - UTILITY_LAMBDA * tk for s, tk in zip(scores, tokens)]

    routing   = compute_routing_metrics(actions, req)

    avg_score   = float(np.mean(scores))
    avg_tokens  = float(np.mean(tokens))
    avg_latency = float(np.mean(latencies))

    # Utility = Q - λC
    Q       = avg_score / 10.0
    C       = avg_tokens / 1000.0
    utility = Q - UTILITY_LAMBDA * C

    return {
        "avg_reward":             float(np.mean(rewards)),
        "avg_judge_score":        avg_score,
        "retrieval_rate":         float(np.mean(actions)),
        "avg_tokens":             avg_tokens,
        "avg_latency":            avg_latency,
        "utility":                utility,
        **routing,
    }


def _per_scenario_breakdown(all_turns: list[dict]) -> dict[str, dict]:
    """
    Group turns by scenario_type and compute per-bucket metrics.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for t in all_turns:
        buckets[t.get("scenario_type", "unknown")].append(t)

    breakdown = {}
    for scenario, turns in buckets.items():
        metrics = _aggregate_turns(turns)
        metrics["n_turns"] = len(turns)
        breakdown[scenario] = metrics
    return breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint_artifacts(
    save_dir: str | Path,
    step: int,
    metrics: dict,
    sample_turns: list[dict] | None = None,
) -> Path:
    """Save metrics.json and sample_predictions.json at a training checkpoint."""
    ckpt_dir = Path(save_dir) / f"checkpoint_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, **metrics}, f, indent=2)

    if sample_turns:
        samples = random.sample(sample_turns, min(10, len(sample_turns)))
        with open(ckpt_dir / "sample_predictions.json", "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info("Checkpoint artifacts saved → %s", ckpt_dir)
    return ckpt_dir


# ──────────────────────────────────────────────────────────────────────────────
# EvaluationCallback
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationCallback(BaseCallback):
    """
    Deterministic validation pass every `eval_every_n_updates` PPO rollouts.

    Logged to TensorBoard and trace_event (PDF p.2 "minimal metric set"):
      • eval/avg_reward
      • eval/avg_judge_score
      • eval/retrieval_rate
      • eval/unsafe_skip_rate        (FN / total)
      • eval/wasteful_retrieve_rate  (FP / total)
      • eval/utility                 (Q - λC)
      • eval/f1
      • eval_scenario/<type>/avg_judge_score  (per scenario_type)
      • eval_scenario/<type>/retrieval_rate
      • eval_scenario/<type>/unsafe_skip_rate
    """

    def __init__(
        self,
        val_episodes: list[list[dict]],
        eval_every_n_updates: int = 5,
        save_path: str = "policy_checkpoints",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.val_episodes          = val_episodes
        self.eval_every_n_updates  = eval_every_n_updates
        self.save_path             = save_path
        self._ppo_updates          = 0
        self._judge                = None
        self._app                  = None

    def _lazy_init(self):
        if self._judge is None:
            from rl_core.utils import JudgeChain
            from core.graph import build_graph
            self._judge = JudgeChain()
            self._app   = build_graph()

    def _on_rollout_end(self) -> None:
        """Called at end of each rollout = 1 PPO update cycle"""
        self._ppo_updates += 1
        if self._ppo_updates % self.eval_every_n_updates != 0:
            return

        self._lazy_init()

        all_turns: list[dict] = []
        for ep in self.val_episodes:
            res = run_episode(
                ep,
                app=self._app,
                judge=self._judge,
                use_policy=True,
                model=self.model,
                collect_samples=True,
            )
            all_turns.extend(res["turns"])

        metrics   = _aggregate_turns(all_turns)
        breakdown = _per_scenario_breakdown(all_turns)

        # ── TensorBoard ───────────────────────────────────────────────────
        self.logger.record("eval/avg_reward",             metrics["avg_reward"])
        self.logger.record("eval/avg_judge_score",        metrics["avg_judge_score"])
        self.logger.record("eval/retrieval_rate",         metrics["retrieval_rate"])
        self.logger.record("eval/unsafe_skip_rate",       metrics["unsafe_skip_rate"])
        self.logger.record("eval/wasteful_retrieve_rate", metrics["wasteful_retrieve_rate"])
        self.logger.record("eval/utility",                metrics["utility"])
        self.logger.record("eval/f1",                     metrics["f1"])

        for scenario, sm in breakdown.items():
            pfx = f"eval_scenario/{scenario}"
            self.logger.record(f"{pfx}/avg_judge_score", sm.get("avg_judge_score", 0))
            self.logger.record(f"{pfx}/retrieval_rate",  sm.get("retrieval_rate",  0))
            self.logger.record(f"{pfx}/unsafe_skip_rate", sm.get("unsafe_skip_rate", 0))

        # ── Clean terminal checkpoint summary ──────────────────────────────
        _hr = "─" * 62
        _ts = time.strftime("%H:%M:%S")
        print(f"\n{_hr}")
        print(
            f"  Validation Checkpoint  │  "
            f"update={self._ppo_updates}  step={self.num_timesteps}  ({_ts})"
        )
        print(_hr)
        print(
            f"  Judge Score    {metrics['avg_judge_score']:5.2f}/10"
            f"     Retrieval Rate    {metrics['retrieval_rate'] * 100:5.1f}%"
        )
        print(
            f"  Unsafe Skip    {metrics['unsafe_skip_rate']:8.3f}"
            f"   Wasteful Retr     {metrics['wasteful_retrieve_rate']:8.3f}"
        )
        print(
            f"  Utility        {metrics['utility']:8.4f}"
            f"   F1               {metrics['f1']:8.3f}"
        )
        print(_hr + "\n")

        # ── trace_event ─────────────────────
        trace_event(
            "VALIDATION_CHECKPOINT",
            ppo_update=self._ppo_updates,
            env_steps=self.num_timesteps,
            avg_reward=metrics["avg_reward"],
            avg_judge_score=metrics["avg_judge_score"],
            retrieval_rate_pct=metrics["retrieval_rate"] * 100,
            unsafe_skip_rate=metrics["unsafe_skip_rate"],
            wasteful_retrieve_rate=metrics["wasteful_retrieve_rate"],
            utility=metrics["utility"],
            f1=metrics["f1"],
        )

        # ── Save checkpoint artifacts ──────────────────────────────────────
        sample_turns = [t for t in all_turns if t.get("response")]
        save_checkpoint_artifacts(
            self.save_path,
            self.num_timesteps,
            {**metrics, "scenario_breakdown": breakdown},
            sample_turns=sample_turns,
        )

    def _on_step(self) -> bool:
        return True


# ──────────────────────────────────────────────────────────────────────────────
# RetrievalStatsCallback  (lightweight rollout-level stats)
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalStatsCallback(BaseCallback):
    """
    Lightweight per-step stats logged every 500 env steps.
    Complements the heavier EvaluationCallback without re-running the judge.
    """

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
# Full evaluate()
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: PPO,
    val_episodes: list[list[dict]],
    beta: float | None = None,
    save_path: str = "policy_checkpoints",
) -> dict[str, Any]:
    """
    Full evaluation comparing 4 strategies on the SAME episodes.

    Strategies:
      a) Policy          — PPO router, deterministic
      b) Always-Retrieve — RAG every turn
      c) Always-Skip     — no retrieval
      d) Heuristic-Router — retrieve on first_turn / topic_shift

    Per strategy (PDF p.7-12):
      avg_judge_score, retrieval_rate, unsafe_skip_rate, wasteful_retrieve_rate,
      avg_tokens, utility (Q - λC), precision, recall, f1, accuracy

    Per-scenario breakdown (PDF p.8): grouped by scenario_type

    Saves (PDF p.4):
      final_evaluation/metrics.json
      final_evaluation/sample_predictions.json  (10 random Policy turns)

    Prints summary table matching PDF p.13.
    """
    from rl_core.utils import JudgeChain
    from core.graph import build_graph

    if beta is None:
        beta = CONFIG.rl_beta

    judge = JudgeChain()
    app   = build_graph()

    strategies = [
        ("Policy",           dict(use_policy=True,   model=model)),
        ("Always-Retrieve",  dict(forced_action=1)),
        ("Always-Skip",      dict(forced_action=0)),
        ("Heuristic-Router", dict(use_heuristic=True)),
    ]

    all_results: dict[str, Any] = {}
    policy_sample_turns: list[dict] = []

    for strategy_name, kwargs in strategies:
        logger.info("Evaluating: %s ...", strategy_name)
        all_turns: list[dict] = []

        for ep in val_episodes:
            res = run_episode(
                ep,
                app=app,
                judge=judge,
                collect_samples=(strategy_name == "Policy"),
                **kwargs,
            )
            all_turns.extend(res["turns"])
            if strategy_name == "Policy":
                policy_sample_turns.extend(res["turns"])

        metrics   = _aggregate_turns(all_turns)
        breakdown = _per_scenario_breakdown(all_turns)
        all_results[strategy_name] = {**metrics, "scenario_breakdown": breakdown}

        trace_event(
            "EVALUATION_STRATEGY",
            strategy=strategy_name,
            avg_judge_score=metrics["avg_judge_score"],
            retrieval_rate_pct=metrics["retrieval_rate"] * 100,
            unsafe_skip_rate=metrics["unsafe_skip_rate"],
            wasteful_retrieve_rate=metrics["wasteful_retrieve_rate"],
            avg_tokens=metrics["avg_tokens"],
            utility=metrics["utility"],
            f1=metrics["f1"],
        )
        print(
            f"  {strategy_name:<20}  "
            f"judge={metrics['avg_judge_score']:.2f}/10  "
            f"ret={metrics['retrieval_rate'] * 100:.0f}%  "
            f"F1={metrics['f1']:.3f}  "
            f"utility={metrics['utility']:.4f}"
        )

    # ── Token savings vs Always-Retrieve baseline ─────────────────────────
    baseline_tokens  = all_results["Always-Retrieve"]["avg_tokens"]
    policy_tokens    = all_results["Policy"]["avg_tokens"]
    tokens_saved     = baseline_tokens - policy_tokens
    tokens_saved_pct = (tokens_saved / baseline_tokens * 100) if baseline_tokens > 0 else 0.0
    all_results["tokens_saved"]     = tokens_saved
    all_results["tokens_saved_pct"] = tokens_saved_pct

    # ── Per-scenario comparison table ─────────────────────────────────────
    scenario_table = _build_scenario_table(all_results)
    all_results["scenario_table"] = scenario_table

    # ── Save final artifacts ─────────────────────────────────────
    final_dir = Path(save_path) / "final_evaluation"
    final_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json — drop the non-serializable scenario_table key first
    to_save = {k: v for k, v in all_results.items() if k != "scenario_table"}
    with open(final_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)

    # sample_predictions.json — 10 random Policy turns
    samples = random.sample(policy_sample_turns, min(10, len(policy_sample_turns)))
    with open(final_dir / "sample_predictions.json", "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    logger.info("Final evaluation artifacts → %s", final_dir)

    # ── Print summary table ────────────────────────────────────
    _print_summary_table(all_results)

    trace_event(
        "EVALUATION_SUMMARY",
        policy_avg_score=all_results["Policy"]["avg_judge_score"],
        policy_retrieval_rate_pct=all_results["Policy"]["retrieval_rate"] * 100,
        policy_avg_tokens=all_results["Policy"]["avg_tokens"],
        policy_utility=all_results["Policy"]["utility"],
        policy_f1=all_results["Policy"]["f1"],
        baseline_avg_score=all_results["Always-Retrieve"]["avg_judge_score"],
        baseline_avg_tokens=all_results["Always-Retrieve"]["avg_tokens"],
        tokens_saved=tokens_saved,
        tokens_saved_pct=tokens_saved_pct,
    )

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Table builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_scenario_table(all_results: dict) -> dict[str, dict]:
    """
    Nested table: scenario_type → strategy → metrics dict.
    PDF p.8 example table format.
    """
    table: dict[str, dict] = {}
    for strategy_name, res in all_results.items():
        if not isinstance(res, dict):
            continue
        breakdown = res.get("scenario_breakdown", {})
        for scenario, sm in breakdown.items():
            table.setdefault(scenario, {})[strategy_name] = {
                "avg_judge_score":        sm.get("avg_judge_score", 0),
                "retrieval_rate":         sm.get("retrieval_rate",  0),
                "unsafe_skip_rate":       sm.get("unsafe_skip_rate", 0),
                "wasteful_retrieve_rate": sm.get("wasteful_retrieve_rate", 0),
                "utility":                sm.get("utility", 0),
                "n_turns":                sm.get("n_turns",  0),
            }
    return table


def _print_summary_table(all_results: dict):
    """
    Print the PDF p.13 style comparison table and per-scenario breakdown.
    """
    strategies = ["Policy", "Always-Retrieve", "Always-Skip", "Heuristic-Router"]
    cols = [
        "avg_judge_score", "retrieval_rate", "avg_tokens",
        "unsafe_skip_rate", "wasteful_retrieve_rate", "utility", "f1",
    ]

    print("\n" + "═" * 108)
    print("  EVALUATION SUMMARY  (RL_Research_Evaluation.pdf — Final Results, p.13)")
    print("═" * 108)
    header = f"{'Strategy':<22}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    print("─" * 108)
    for strat in strategies:
        if strat not in all_results:
            continue
        res = all_results[strat]
        row = f"{strat:<22}"
        for c in cols:
            v = res.get(c, float("nan"))
            if c == "retrieval_rate":
                row += f"{v * 100:>17.1f}%"
            elif c == "avg_tokens":
                row += f"{v:>18.0f}"
            else:
                row += f"{v:>18.4f}"
        print(row)

    print("─" * 108)
    print(
        f"  Tokens saved (Policy vs Always-Retrieve): "
        f"{all_results.get('tokens_saved', 0):.0f} "
        f"({all_results.get('tokens_saved_pct', 0):.1f}%)"
    )
    print("═" * 108)

    # Per-scenario table (Policy only)
    scenario_table = all_results.get("scenario_table", {})
    if scenario_table:
        print("\n  PER-SCENARIO BREAKDOWN  (Policy)  — PDF p.8")
        print("─" * 82)
        print(
            f"{'Scenario':<22}{'n':>6}{'judge':>9}{'ret%':>9}"
            f"{'unsafe_skip':>13}{'wasteful':>11}{'utility':>11}"
        )
        print("─" * 82)
        for scenario, by_strat in sorted(scenario_table.items()):
            if "Policy" not in by_strat:
                continue
            sm = by_strat["Policy"]
            print(
                f"{scenario:<22}{sm['n_turns']:>6}{sm['avg_judge_score']:>9.2f}"
                f"{sm['retrieval_rate'] * 100:>8.1f}%"
                f"{sm['unsafe_skip_rate']:>13.3f}"
                f"{sm['wasteful_retrieve_rate']:>11.3f}"
                f"{sm['utility']:>11.4f}"
            )
        print("═" * 82)

    # Confusion-matrix summary for Policy
    if "Policy" in all_results:
        p = all_results["Policy"]
        print("\n  CONFUSION MATRIX  (Policy)  — PDF p.10")
        print("─" * 42)
        print(f"  {'':30} Retrieval Needed   Skip OK")
        print(f"  {'Retrieve':30} {'TP='+str(p.get('TP','?')):18} {'FP='+str(p.get('FP','?'))}")
        print(f"  {'Skip':30} {'FN='+str(p.get('FN','?')):18} {'TN='+str(p.get('TN','?'))}")
        print("─" * 42)
        print(
            f"  Accuracy={p.get('accuracy', 0):.3f}  "
            f"Precision={p.get('precision', 0):.3f}  "
            f"Recall={p.get('recall', 0):.3f}  "
            f"F1={p.get('f1', 0):.3f}"
        )
        print("═" * 42)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    train_episodes: list[list[dict]],
    val_episodes: list[list[dict]],
    total_timesteps: int = 20_000,
    save_path: str = "policy_checkpoints",
    tensorboard_log: str = "./tb_logs",
    trace_log: str = "./training_logs/training_trace.log",
):
    """
    Train PPO policy. PPO hyperparameters are UNCHANGED from original.

    EvaluationCallback fires every ~total_timesteps/10/n_steps PPO updates,
    logging the full metric set from the PDF to TensorBoard + trace file.

    At end of training, automatically runs the full 4-baseline evaluate().
    """
    trace_log_path = configure_training_trace(trace_log)

    # ── Training banner ───────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print("  PPO Retrieval Policy  –  Training")
    print("─" * 64)
    print(f"  Timesteps     : {total_timesteps:,}")
    print(f"  Train eps     : {len(train_episodes)}   |   Val eps : {len(val_episodes)}")
    print(f"  Save path     : {save_path}")
    print(f"  Trace log     : {trace_log_path}")
    print("═" * 64 + "\n")

    trace_event(
        "TRAINING_START",
        total_timesteps=total_timesteps,
        train_episodes=len(train_episodes),
        val_episodes=len(val_episodes),
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
        learning_rate=1e-3,    # high LR — only the tiny MLP head is trained
        n_steps=256,           # rollout buffer; n_steps % batch_size must == 0
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,         # encourage exploration (small action space)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=0,   # our callbacks provide terminal output
    )

    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    # ── EvaluationCallback: ~10 eval points across total training  (PDF p.6) ─
    # n_steps=256 → 1 update per 256 env steps
    # We want ~10 eval checkpoints → eval every total_timesteps / (256 * 10) updates
    updates_per_eval = max(1, total_timesteps // (256 * 10))

    print(f"  PPO model ready  |  trainable params: {trainable_params:,}")
    print(f"  Eval checkpoints : every {updates_per_eval} update(s)  (~{total_timesteps // (updates_per_eval * 256)} checkpoints total)\n")

    eval_callback  = EvaluationCallback(
        val_episodes=val_episodes,
        eval_every_n_updates=updates_per_eval,
        save_path=save_path,
        verbose=1,
    )
    stats_callback = RetrievalStatsCallback()

    # ── Train ─────────────────────────────────────────────────────────────
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

    # ── Automatic full post-training evaluation with all 4 baselines ────────
    print("\n" + "═" * 64)
    print("  POST-TRAINING EVALUATION  –  4 Strategies")
    print("─" * 64 + "\n")
    results = evaluate(model, val_episodes, save_path=save_path)
    return model, results


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train retrieval-decision policy with PPO")
    p.add_argument(
        "--episodes",
        default="dataset_generator/dialog_dataset/conversations.json",
        help="Path to conversations JSON file",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=20_000,
        help="Total PPO training timesteps",
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_episodes = load_episodes(args.episodes)
    train_eps, val_eps = split_episodes(all_episodes)

    if args.eval_only:
        logger.info("Eval-only mode: loading policy from %s", args.policy_path)
        configure_training_trace(args.trace_log)
        model = PPO.load(args.policy_path)
        results = evaluate(model, val_eps, save_path=args.save_path)
    else:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        model, results = train(
            train_episodes=train_eps,
            val_episodes=val_eps,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            tensorboard_log=args.tb_log,
            trace_log=args.trace_log,
        )

    # Full summary table printed inside evaluate(); nothing more needed here
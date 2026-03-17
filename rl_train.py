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
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from core.config import CONFIG
from rl_core.evaluation import EvaluationCallback, evaluate
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
    logging the full metric set to TensorBoard + trace file.

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

    # ── EvaluationCallback: ~10 eval points across total training ─
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
    print("\n" + "═" * 80)
    print("  POST-TRAINING EVALUATION  –  4 Strategies")
    print("  Comparing: Policy (RL) vs Always-Retrieve vs Always-Skip vs Heuristic-Router")
    print("─" * 80)
    print(f"  Evaluating on {len(val_episodes)} validation episodes...")
    print(f"  Results will be saved to: {save_path}/final_evaluation/")
    print("═" * 80 + "\n")

    t_eval_start = time.time()
    results = evaluate(model, val_episodes, save_path=save_path)
    t_eval_duration = time.time() - t_eval_start

    print("\n" + "═" * 80)
    print(f"  POST-TRAINING EVALUATION COMPLETE  ({t_eval_duration:.1f}s)")
    print("═" * 80 + "\n")
    
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
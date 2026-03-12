"""
rl_train.py
───────────
Train and evaluate the retrieval-decision policy with PPO.

Usage
─────
  # Full training run:
  python rl_train.py --episodes conversations.json --timesteps 20000

  # Quick smoke-test (fewer steps):
  python rl_train.py --episodes conversations.json --timesteps 500 --eval-only

  # Eval only (load saved policy):
  python rl_train.py --episodes conversations.json --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from config import CONFIG
from rl_env import RAGDecisionEnv
from rl_policy import TransformerHeadPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rl_train")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_episodes(path: str) -> list[list[dict]]:
    """
    Load conversations.json.
    Expected format: list of episode objects, each with a "turns" key.
    Returns: list of turn lists  [[{query, ground_truth}, ...], ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    episodes: list[list[dict]] = []
    for ep in data:
        turns = ep.get("turns", ep)  # support both {turns:[...]} and [[...]]
        episodes.append(
            [{"query": t["query"], "ground_truth": t["ground_truth"]} for t in turns]
        )
    logger.info("Loaded %d episodes", len(episodes))
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
    # Ensure at least one episode in val even for tiny datasets
    if not val:
        val = [train[-1]]
    logger.info("Train episodes: %d | Val episodes: %d", len(train), len(val))
    return train, val


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalStatsCallback(BaseCallback):
    """
    Logs retrieval rate and mean reward to tensorboard every N rollouts.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._actions: list[int] = []
        self._rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])

        for act, info in zip(actions, infos):
            self._actions.append(int(act))
            if "score" in info:
                self._rewards.append(float(info["score"]))

        if self.n_calls % 500 == 0 and self._actions:
            retrieve_rate = sum(self._actions) / len(self._actions)
            mean_reward = (
                float(np.mean(self._rewards)) if self._rewards else float("nan")
            )
            self.logger.record("custom/retrieval_rate", retrieve_rate)
            self.logger.record("custom/mean_judge_score", mean_reward)
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
    save_path: str = "policy_model",
    tensorboard_log: str = "./tb_logs",
):
    logger.info("Building training environment...")
    train_env = DummyVecEnv([lambda: RAGDecisionEnv(train_episodes, mode="train")])
    eval_env = DummyVecEnv([lambda: RAGDecisionEnv(val_episodes, mode="eval")])

    # ── PPO configuration ────────────────────────────────────────────────────
    # LR is higher than typical because ONLY the tiny MLP head is updated.
    model = PPO(
        policy=TransformerHeadPolicy,
        env=train_env,
        learning_rate=1e-3,          # high LR — only head is trained
        n_steps=256,                 # rollout buffer size per env; must satisfy n_steps % batch_size == 0
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,               # encourage exploration (small action space)
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tensorboard_log,
        verbose=1,
    )

    logger.info(
        "PPO model created. Trainable params: %d",
        sum(p.numel() for p in model.policy.parameters() if p.requires_grad),
    )

    # ── Callbacks ────────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=max(1, total_timesteps // 10),
        n_eval_episodes=len(val_episodes),
        deterministic=True,
        render=False,
    )
    stats_callback = RetrievalStatsCallback()

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Starting PPO training for %d timesteps...", total_timesteps)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, stats_callback],
        tb_log_name="rag_retrieval_policy",
        progress_bar=True,
    )

    final_path = f"{save_path}/policy_model_final"
    model.save(final_path)
    logger.info("Final model saved to %s.zip", final_path)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: PPO,
    val_episodes: list[list[dict]],
    beta: float | None = None,
):
    """
    Run evaluation on val_episodes and compare:
      - Policy mode  (model.predict deterministic)
      - Baseline     (always retrieve, action=1)

    Logs:
      avg_reward, avg_score, retrieval_%, tokens_saved vs baseline
    """
    from utils import JudgeChain, approx_token_count
    from state_encoder import encode_state
    from langchain_core.messages import HumanMessage
    from graph import build_graph
    from observability import build_run_config
    import random

    if beta is None:
        beta = CONFIG.rl_beta

    judge = JudgeChain()
    app = build_graph()

    def run_episode(episode, forced_action: int | None = None, use_policy=False):
        summary = ""
        thread_id = f"eval-{random.randint(0, 10_000_000)}"
        run_config = build_run_config(thread_id)
        total_score = 0.0
        total_tokens = 0
        retrieval_count = 0

        for turn in episode:
            query = turn["query"]
            gt = turn["ground_truth"]

            if use_policy:
                obs = encode_state(summary, query)
                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(action_arr)
            else:
                action = forced_action  # type: ignore

            input_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "summary": summary,
                "retrieved_docs": [],
                "answer": "",
                "mode": "baseline",
            }
            run_config["configurable"]["forced_action"] = action  # per-turn, not persisted in state

            final_state: dict = {}
            for event in app.stream(input_state, config=run_config):
                for _, partial in event.items():
                    if partial:
                        final_state.update(partial)

            response = final_state.get("answer", "")
            docs = final_state.get("retrieved_docs", [])
            summary = final_state.get("summary", summary)

            result = judge.invoke(
                {"query": query, "response": response, "ground_truth": gt}
            )
            score = float(result.get("score", 5))
            tokens = approx_token_count(docs)
            total_score += score
            total_tokens += tokens
            if action == 1:
                retrieval_count += 1

        n = len(episode)
        return {
            "avg_score": total_score / n,
            "total_tokens": total_tokens,
            "retrieval_rate": retrieval_count / n,
        }

    logger.info("=== Evaluation: Policy vs Baseline ===")
    policy_scores, policy_tokens, policy_ret = [], [], []
    baseline_scores, baseline_tokens = [], []

    for ep in val_episodes:
        p = run_episode(ep, use_policy=True)
        b = run_episode(ep, forced_action=1)
        policy_scores.append(p["avg_score"])
        policy_tokens.append(p["total_tokens"])
        policy_ret.append(p["retrieval_rate"])
        baseline_scores.append(b["avg_score"])
        baseline_tokens.append(b["total_tokens"])

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    tokens_saved = _mean(baseline_tokens) - _mean(policy_tokens)
    tokens_saved_pct = (
        tokens_saved / _mean(baseline_tokens) * 100 if _mean(baseline_tokens) > 0 else 0
    )

    logger.info("Policy   — avg_score: %.3f | retrieval%%: %.1f%% | avg_tokens: %.0f",
                _mean(policy_scores), _mean(policy_ret) * 100, _mean(policy_tokens))
    logger.info("Baseline — avg_score: %.3f | retrieval%%: 100%%   | avg_tokens: %.0f",
                _mean(baseline_scores), _mean(baseline_tokens))
    logger.info(
        "Tokens saved vs baseline: %.0f (%.1f%%)", tokens_saved, tokens_saved_pct
    )

    return {
        "policy_avg_score": _mean(policy_scores),
        "policy_retrieval_rate": _mean(policy_ret),
        "policy_avg_tokens": _mean(policy_tokens),
        "baseline_avg_score": _mean(baseline_scores),
        "baseline_avg_tokens": _mean(baseline_tokens),
        "tokens_saved": tokens_saved,
        "tokens_saved_pct": tokens_saved_pct,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train retrieval-decision policy with PPO")
    p.add_argument(
        "--episodes",
        default="conversations.json",
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    all_episodes = load_episodes(args.episodes)
    train_eps, val_eps = split_episodes(all_episodes)

    if args.eval_only:
        logger.info("Eval-only mode: loading policy from %s", args.policy_path)
        model = PPO.load(args.policy_path)
    else:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
        model = train(
            train_episodes=train_eps,
            val_episodes=val_eps,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            tensorboard_log=args.tb_log,
        )

    results = evaluate(model, val_eps)
    print("\n── Evaluation Summary ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")
    print("────────────────────────────────────────────────────\n")
"""
rl_env.py
─────────
Gymnasium environment for training the retrieval-decision policy.

Observation  : float32 vector of shape (encoder_dim,)
               = mean-pooled transformer embedding of "[summary] [SEP] [query]"

Action space : Discrete(2)
               0 = SKIP retrieval
               1 = RETRIEVE

Reward       : judge_score  -  β × approx_token_count(retrieved_docs)

Episode      : one multi-turn conversation from the training dataset.
               Each step = one conversation turn.
               Episode ends when all turns are exhausted.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from langchain_core.messages import HumanMessage

from config import CONFIG
from graph import build_graph
from observability import build_run_config
from state_encoder import encode_state
from utils import JudgeChain, approx_token_count

logger = logging.getLogger("disaster_chatbot")


class RAGDecisionEnv(gym.Env):
    """
    Each reset() samples a random episode (multi-turn conversation).
    Each step(action) runs ONE turn through the LangGraph with the forced action,
    obtains the answer, scores it via the judge, and returns the next observation.

    Parameters
    ----------
    episodes : list of episodes, where each episode is a list of turn dicts:
               [{"query": str, "ground_truth": str}, ...]
    beta     : float — retrieval cost coefficient (default: CONFIG.rl_beta)
    mode     : "train" (random episode sampling) | "eval" (sequential)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        episodes: list[list[dict]],
        beta: float | None = None,
        mode: str = "train",
    ):
        super().__init__()

        self.episodes = episodes
        self.beta = beta if beta is not None else CONFIG.rl_beta
        self.mode = mode

        # ── Spaces ────────────────────────────────────────────────────────────
        encoder_dim = CONFIG.policy_encoder_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(encoder_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)  # 0=skip, 1=retrieve

        # ── Runtime state ─────────────────────────────────────────────────────
        self._episode_idx: int = 0          # for sequential eval
        self._current_episode: list[dict] = []
        self._turn_idx: int = 0
        self._summary: str = ""
        self._thread_id: str = "rl-env-default"
        self._graph = build_graph()
        self._judge = JudgeChain()

        # Stats for debugging
        self._episode_rewards: list[float] = []

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        # Sample episode
        if self.mode == "train":
            self._current_episode = random.choice(self.episodes)
        else:
            self._current_episode = self.episodes[
                self._episode_idx % len(self.episodes)
            ]
            self._episode_idx += 1

        self._turn_idx = 0
        self._summary = ""
        self._thread_id = f"rl-env-{random.randint(0, 10_000_000)}"
        self._episode_rewards = []

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):
        turn = self._current_episode[self._turn_idx]
        query: str = turn["query"]
        ground_truth: str = turn["ground_truth"]

        # ── Run one graph turn with forced action ─────────────────────────────
        _action = int(action)  # cast numpy.int64 → int for msgpack serialisation
        input_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "summary": self._summary,
            "retrieved_docs": [],
            "answer": "",
            "mode": "baseline",
        }
        run_config = build_run_config(self._thread_id)
        run_config["configurable"]["forced_action"] = _action  # per-invocation, not persisted in state

        final_state: dict = {}
        try:
            for event in self._graph.stream(input_state, config=run_config):
                for _, partial in event.items():
                    if partial:
                        final_state.update(partial)
        except Exception as exc:
            logger.error("[env.step] graph error at turn %d: %s", self._turn_idx, exc)
            # Return zero reward and end episode on graph failure
            obs = self._build_obs()
            return obs, 0.0, True, False, {"error": str(exc)}

        response: str = final_state.get("answer", "")
        retrieved_docs = final_state.get("retrieved_docs", [])
        self._summary = final_state.get("summary", self._summary)

        # ── Reward ────────────────────────────────────────────────────────────
        judge_result = self._judge.invoke(
            {
                "query": query,
                "response": response,
                "ground_truth": ground_truth,
            }
        )
        score: float = float(judge_result.get("score", 5))
        token_cost: float = self.beta * approx_token_count(retrieved_docs)
        reward: float = score - token_cost

        logger.debug(
            "[env.step] turn=%d action=%d score=%.1f token_cost=%.3f reward=%.3f",
            self._turn_idx, action, score, token_cost, reward,
        )
        self._episode_rewards.append(reward)

        # ── Advance turn ──────────────────────────────────────────────────────
        self._turn_idx += 1
        done = self._turn_idx >= len(self._current_episode)

        if done:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            logger.info(
                "[env] episode done | turns=%d avg_reward=%.3f",
                len(self._current_episode),
                avg_reward,
            )

        obs = self._build_obs()
        info: dict[str, Any] = {
            "turn": self._turn_idx,
            "action": action,
            "score": score,
            "token_cost": token_cost,
            "judge_reason": judge_result.get("reason", ""),
        }
        return obs, reward, done, False, info

    def render(self):
        pass  # no rendering

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Build the observation for the CURRENT (not yet executed) turn."""
        if self._turn_idx < len(self._current_episode):
            query = self._current_episode[self._turn_idx]["query"]
        else:
            query = ""  # episode is over — dummy obs
        return encode_state(self._summary, query)
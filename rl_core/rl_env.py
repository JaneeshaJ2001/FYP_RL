"""
rl_env.py
─────────
Gymnasium environment for training the retrieval-decision policy.

Observation  : float32 vector of shape (encoder_dim,)
               = mean-pooled transformer embedding of a richer context string:
               "[TURN <N>] [SEP] [PREV_ACTION <label>] [SEP] [summary] [SEP] [query]"

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

from core.config import CONFIG
from core.graph import build_graph
from core.nodes import (
    _format_docs as _format_retrieved_docs,
    INITIAL_GROUNDED_SUMMARY,          
)
from core.observability import build_run_config
from rl_core.state_encoder import encode_state
from rl_core.training_trace import trace_event
from rl_core.utils import JudgeChain, approx_token_count

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
        self._episode_idx: int = 0
        self._current_episode: list[dict] = []
        self._turn_idx: int = 0
        self._summary: str = INITIAL_GROUNDED_SUMMARY  
        self._thread_id: str = "rl-env-default"
        self._graph = build_graph()
        self._judge = JudgeChain()    
        self._turn_number: int = 1   # 1-based turn counter for current episode
        self._prev_action: int = -1  # -1 = no previous turn (first turn sentinel)

        # Stats for debugging
        self._episode_rewards: list[float] = []
        self._episode_actions: list[int] = []
        self._episode_counter: int = 0

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
    
        self._summary = INITIAL_GROUNDED_SUMMARY  # canonical first-turn sentinel
        self._turn_number = 1                     # reset 1-based turn counter
        self._prev_action = -1                    # no previous action yet
        # ──────────────────────────────────────────────────────────────────────
        self._thread_id = f"rl-env-{random.randint(0, 10_000_000)}"
        self._episode_rewards = []
        self._episode_actions = []
        self._episode_counter += 1

        trace_event(
            "EPISODE_START",
            mode=self.mode,
            episode=self._episode_counter,
            turns=len(self._current_episode),
        )

        obs = self._build_obs()
        return obs, {}

    def step(self, action: int):
        turn = self._current_episode[self._turn_idx]
        query: str = turn["query"]

        _action = int(action)  # cast numpy.int64 → int for msgpack serialisation

        # ── Run one graph turn with forced action ─────────────────────────────
        input_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "summary": self._summary,
            "retrieved_docs": [],
            "answer": "",
            "mode": "rl",        
            "turn_number": self._turn_number,
            "prev_action": self._prev_action,
        }
        run_config = build_run_config(self._thread_id)
        run_config["configurable"]["forced_action"] = _action

        final_state: dict = {}
        try:
            for event in self._graph.stream(input_state, config=run_config):
                for _, partial in event.items():
                    if partial:
                        final_state.update(partial)
        except Exception as exc:
            logger.error("[env.step] graph error at turn %d: %s", self._turn_idx, exc)
            trace_event(
                "GRAPH_ERROR",
                mode=self.mode,
                episode=self._episode_counter,
                turn=f"{self._turn_idx + 1}/{len(self._current_episode)}",
                error=str(exc),
            )
            obs = self._build_obs()
            return obs, 0.0, True, False, {"error": str(exc)}

        response: str = final_state.get("answer", "")
        retrieved_docs = final_state.get("retrieved_docs", [])
    
        # Read back the updated grounded summary and new counters from the graph
        self._summary      = final_state.get("summary", self._summary)
        self._turn_number  = final_state.get("turn_number", self._turn_number + 1)
        self._prev_action  = _action   # the action just executed becomes prev_action

        # ── Reward ────────────────────────────────────────────────────────────
        docs_text = _format_retrieved_docs(retrieved_docs) if retrieved_docs else ""
        judge_result = self._judge.invoke(
            {
                "query": query,
                "conversation_summary": self._summary,
                "action_taken": "retrieve" if _action == 1 else "skip",
                "retrieved_docs": docs_text,
                "response": response,
            }
        )
        score: float = float(judge_result.get("score", 5))
        token_cost: float = self.beta * approx_token_count(retrieved_docs)
        reward: float = score - token_cost

        action_label = "RETRIEVE" if _action == 1 else "SKIP"
        trace_event(
            "TURN",
            mode=self.mode,
            episode=self._episode_counter,
            turn=f"{self._turn_idx + 1}/{len(self._current_episode)}",
            action=action_label,        
            turn_number=self._turn_number - 1,   # log the turn that just completed
            prev_action=self._prev_action,
            # ──────────────────────────────────────────────────────────────────
            score=score,
            token_cost=token_cost,
            reward=reward,
            docs=len(retrieved_docs),
            query=query,
            response=response,
            judge=judge_result.get("reason", ""),
        )
        self._episode_rewards.append(reward)
        self._episode_actions.append(_action)

        # ── Advance turn ──────────────────────────────────────────────────────
        self._turn_idx += 1
        done = self._turn_idx >= len(self._current_episode)

        if done:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            ret_rate_pct = sum(self._episode_actions) / len(self._episode_actions) * 100
            trace_event(
                "EPISODE_END",
                mode=self.mode,
                episode=self._episode_counter,
                turns=len(self._current_episode),
                avg_reward=avg_reward,
                retrieval_rate_pct=ret_rate_pct,
            )
            logger.info(
                "[env] episode done | turns=%d avg_reward=%.3f retrieval_rate=%.0f%%",
                len(self._current_episode), avg_reward, ret_rate_pct,
            )
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._build_obs()

        info: dict[str, Any] = {
            "turn":         self._turn_idx,
            "action":       action,
            "score":        score,
            "token_cost":   token_cost,
            "judge_reason": judge_result.get("reason", ""),        
            "turn_number":  self._turn_number,
            "prev_action":  self._prev_action,
        }
        return obs, reward, done, False, info

    def render(self):
        pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        Build the observation for the CURRENT (not yet executed) turn.

        === NEW GROUNDED STATE LOGIC ===
        Passes turn_number and prev_action to encode_state so the transformer
        sees the richer contextual prefix.
        """
        if self._turn_idx < len(self._current_episode):
            query = self._current_episode[self._turn_idx]["query"]
        else:
            query = ""   # episode is over — dummy obs

        return encode_state(
            summary=self._summary,
            query=query,
            turn_number=self._turn_number,  
            prev_action=self._prev_action,  
        )
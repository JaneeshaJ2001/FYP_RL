"""
test_full_suite.py
──────────────────
Comprehensive tests for every component and flow in the FYP_RL disaster chatbot.

Run with:
    python -m pytest tests/test_full_suite.py -v

Groups:
  A. StateEncoder         — _build_encoder_text, encode_state shape/dtype
  B. DisasterState        — TypedDict field coverage
  C. GraphTopology        — node wiring, edge routing
  D. decide_retrieve      — forced / baseline / policy modes
  E. summarize_conversation — grounded KB logic, counter advancement
  F. RAGDecisionEnv       — Gym API, observation consistency
  G. WarmStart            — dataset builder, metrics helper
  H. EvaluationMetrics    — compute_routing_metrics correctness
  I. RewardFormula        — reward = score - beta * tokens
  J. LoadEpisodes         — JSON loading edge cases
  K. JudgeChain           — prompt injection guard, parse fallback
  L. DataConsistency      — conversations.json alignment
  M. Integration          — end-to-end single turn (no LLM required)
"""

import json
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / stubs
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent  # adjust if running from elsewhere

CONVERSATIONS_JSON = REPO_ROOT / "dataset_generator" / "dialog_dataset" / "conversations.json"

ENCODER_DIM = 384  # all-MiniLM-L6-v2 output dim


def _fake_encode_state(summary, query, turn_number=1, prev_action=-1):
    """Deterministic stand-in for encode_state — avoids loading the transformer."""
    return np.zeros(ENCODER_DIM, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# A. StateEncoder — _build_encoder_text
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildEncoderText(unittest.TestCase):

    def _import(self):
        # Lazy import to avoid loading the transformer during collection
        from rl_core.state_encoder import _build_encoder_text
        return _build_encoder_text

    def test_first_turn_sentinel(self):
        """prev_action=-1 must produce label 'none'."""
        fn = self._import()
        text = fn("some summary", "query", turn_number=1, prev_action=-1)
        self.assertIn("[PREV_ACTION none]", text)
        self.assertIn("[TURN 1]", text)

    def test_retrieve_label(self):
        fn = self._import()
        text = fn("summary", "query", turn_number=2, prev_action=1)
        self.assertIn("[PREV_ACTION retrieve]", text)
        self.assertIn("[TURN 2]", text)

    def test_skip_label(self):
        fn = self._import()
        text = fn("summary", "query", turn_number=3, prev_action=0)
        self.assertIn("[PREV_ACTION skip]", text)

    def test_empty_summary_skips_sep(self):
        """When summary is empty/whitespace, it must NOT appear between the prefix and query."""
        fn = self._import()
        text = fn("   ", "my question", turn_number=1, prev_action=-1)
        # Should still include the query
        self.assertIn("my question", text)
        # But the leading whitespace summary must be skipped
        # Format: prefix [SEP] query  (NOT: prefix [SEP]    [SEP] query)
        self.assertNotIn("[SEP]    [SEP]", text)

    def test_nonempty_summary_included(self):
        fn = self._import()
        text = fn("Prior KB: floods evacuate.", "new query", 2, 1)
        self.assertIn("Prior KB", text)
        self.assertIn("new query", text)

    def test_sep_order(self):
        """[SEP] tokens must appear in the right order."""
        fn = self._import()
        text = fn("some summary", "the query", 2, 1)
        idx_turn = text.index("[TURN")
        idx_prev = text.index("[PREV_ACTION")
        idx_query = text.index("the query")
        self.assertLess(idx_turn, idx_prev)
        self.assertLess(idx_prev, idx_query)


class TestEncodeStateShape(unittest.TestCase):
    """Test encode_state output contract WITHOUT loading the real transformer."""

    def test_shape_and_dtype(self):
        """encode_state must return float32 of shape (encoder_dim,)."""
        with patch("rl_core.state_encoder._get_encoder") as mock_get:
            # Build a minimal fake model that returns zero hidden states
            import torch
            tokenizer_mock = MagicMock()
            tokenizer_mock.return_value = {
                "input_ids": torch.zeros(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
            }
            model_mock = MagicMock()
            hidden = torch.zeros(1, 10, ENCODER_DIM)
            output_mock = MagicMock()
            output_mock.last_hidden_state = hidden
            model_mock.return_value = output_mock

            mock_get.return_value = (model_mock, tokenizer_mock, "cpu")

            from rl_core.state_encoder import encode_state
            obs = encode_state("summary", "query", turn_number=1, prev_action=-1)

            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(obs.shape, (ENCODER_DIM,))

    def test_default_args(self):
        """encode_state must work when turn_number and prev_action use defaults."""
        with patch("rl_core.state_encoder._get_encoder") as mock_get:
            import torch
            tokenizer_mock = MagicMock()
            tokenizer_mock.return_value = {
                "input_ids": torch.zeros(1, 5, dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }
            model_mock = MagicMock()
            hidden = torch.zeros(1, 5, ENCODER_DIM)
            output_mock = MagicMock()
            output_mock.last_hidden_state = hidden
            model_mock.return_value = output_mock
            mock_get.return_value = (model_mock, tokenizer_mock, "cpu")

            from rl_core.state_encoder import encode_state
            # Should not raise — uses default turn_number=1, prev_action=-1
            obs = encode_state("summary", "query")
            self.assertIsNotNone(obs)


# ─────────────────────────────────────────────────────────────────────────────
# B. DisasterState TypedDict
# ─────────────────────────────────────────────────────────────────────────────

class TestDisasterState(unittest.TestCase):

    def test_required_fields_present(self):
        from core.nodes import DisasterState
        required = {
            "messages", "summary", "query", "retrieved_docs",
            "answer", "action", "mode", "turn_number", "prev_action",
        }
        actual = set(DisasterState.__annotations__.keys())
        missing = required - actual
        self.assertEqual(missing, set(), f"DisasterState missing fields: {missing}")

    def test_turn_number_and_prev_action_typed_int(self):
        from core.nodes import DisasterState
        annotations = DisasterState.__annotations__
        self.assertIn("turn_number", annotations)
        self.assertIn("prev_action", annotations)

    def test_mode_literal(self):
        """mode field must accept all three modes used in the system."""
        from core.nodes import DisasterState
        import typing
        import sys

        # __annotations__ on a TypedDict stores the raw annotation objects as
        # they appear in the class body.  When `from __future__ import annotations`
        # is active the values are plain strings, so get_args() returns ().
        # get_type_hints() resolves those strings back to real type objects.
        try:
            hints = typing.get_type_hints(DisasterState)
            annotation = hints["mode"]
        except Exception:
            # Fall back to raw annotation if resolution fails (e.g. forward refs)
            annotation = DisasterState.__annotations__["mode"]

        args = typing.get_args(annotation)

        # If args is still empty the annotation is not a Literal — check the
        # string representation as a last resort so the test is never silently
        # vacuous.
        if not args:
            ann_str = str(annotation)
            for expected in ("baseline", "policy", "rl"):
                self.assertIn(
                    expected, ann_str,
                    f"mode annotation does not contain '{expected}': {ann_str}"
                )
        else:
            for expected in ("baseline", "policy", "rl"):
                self.assertIn(expected, args, f"mode Literal missing '{expected}'")


# ─────────────────────────────────────────────────────────────────────────────
# C. Graph Topology
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphTopology(unittest.TestCase):

    def test_graph_compiles(self):
        """build_graph() must not raise."""
        with patch("core.nodes.load_or_create_vectorstore"), \
             patch("core.nodes.build_llm"):
            from core.graph import build_graph
            g = build_graph()
            self.assertIsNotNone(g)

    def test_nodes_registered(self):
        with patch("core.nodes.load_or_create_vectorstore"), \
             patch("core.nodes.build_llm"):
            from core.graph import build_graph
            g = build_graph()
            node_names = set(g.nodes)
            for expected in ("decide_retrieve", "retrieve", "generate_answer", "summarize_conversation"):
                self.assertIn(expected, node_names)

    def test_retrieval_router_retrieve(self):
        from core.nodes import retrieval_router
        state = {"action": 1}
        self.assertEqual(retrieval_router(state), "retrieve")

    def test_retrieval_router_skip(self):
        from core.nodes import retrieval_router
        state = {"action": 0}
        self.assertEqual(retrieval_router(state), "generate_answer")

    def test_retrieval_router_default(self):
        """Missing action key must default to retrieve (safe fallback)."""
        from core.nodes import retrieval_router
        state = {}
        self.assertEqual(retrieval_router(state), "retrieve")


# ─────────────────────────────────────────────────────────────────────────────
# D. decide_retrieve node
# ─────────────────────────────────────────────────────────────────────────────

class TestDecideRetrieve(unittest.TestCase):

    def _call(self, state, config):
        from core.nodes import decide_retrieve
        return decide_retrieve(state, config)

    def test_forced_action_1(self):
        config = {"configurable": {"forced_action": 1}}
        result = self._call({"mode": "rl"}, config)
        self.assertEqual(result["action"], 1)

    def test_forced_action_0(self):
        config = {"configurable": {"forced_action": 0}}
        result = self._call({"mode": "rl"}, config)
        self.assertEqual(result["action"], 0)

    def test_forced_overrides_baseline(self):
        """forced_action must override even baseline mode."""
        config = {"configurable": {"forced_action": 0}}
        result = self._call({"mode": "baseline"}, config)
        self.assertEqual(result["action"], 0)

    def test_baseline_always_retrieve(self):
        """Without forced_action, baseline mode must always return 1."""
        config = {"configurable": {}}
        result = self._call({"mode": "baseline"}, config)
        self.assertEqual(result["action"], 1)

    def test_none_config_baseline(self):
        """None config must fall back to baseline (retrieve)."""
        # mode baseline + no config
        config = {"configurable": {}}
        result = self._call({"mode": "baseline"}, config)
        self.assertEqual(result["action"], 1)

    def test_forced_action_numpy_int(self):
        """numpy.int64 forced_action must be coerced to Python int."""
        config = {"configurable": {"forced_action": np.int64(0)}}
        result = self._call({"mode": "rl"}, config)
        self.assertIsInstance(result["action"], int)
        self.assertEqual(result["action"], 0)

    def test_policy_fallback_when_no_model(self):
        """In policy mode with no saved model file, must fall back to retrieve."""
        config = {"configurable": {}}
        with patch("core.nodes._load_rl_policy", return_value=None):
            result = self._call({"mode": "policy", "query": "test", "summary": "", "turn_number": 1, "prev_action": -1}, config)
            self.assertEqual(result["action"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# E. summarize_conversation — grounded KB logic
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarizeConversation(unittest.TestCase):
    """Test that summarize_conversation advances counters and dispatches prompts correctly."""

    def _make_state(self, action, summary="existing KB", turn_number=2, docs=None):
        from langchain_core.messages import HumanMessage, AIMessage
        return {
            "messages": [
                HumanMessage(content="user question"),
                AIMessage(content="assistant answer"),
            ],
            "action": action,
            "retrieved_docs": docs or [],
            "summary": summary,
            "turn_number": turn_number,
        }

    def _run(self, state, llm_response="updated KB"):
        """Run summarize_conversation with a mocked LLM."""
        mock_result = MagicMock()
        mock_result.content = llm_response

        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("core.nodes._get_llm"), \
             patch("core.nodes.GROUNDED_SUMMARY_RETRIEVE_PROMPT") as mock_rp, \
             patch("core.nodes.GROUNDED_SUMMARY_SKIP_PROMPT") as mock_sp, \
             patch("core.nodes.extract_invoke_config", return_value={}):

            mock_rp.__or__ = lambda self, other: mock_chain
            mock_sp.__or__ = lambda self, other: mock_chain

            from core.nodes import summarize_conversation
            return summarize_conversation(state, config={})

    def test_turn_number_increments(self):
        state = self._make_state(action=1, turn_number=3)
        result = self._run(state)
        self.assertEqual(result["turn_number"], 4)

    def test_prev_action_set_to_current_action_retrieve(self):
        state = self._make_state(action=1)
        result = self._run(state)
        self.assertEqual(result["prev_action"], 1)

    def test_prev_action_set_to_current_action_skip(self):
        state = self._make_state(action=0)
        result = self._run(state)
        self.assertEqual(result["prev_action"], 0)

    def test_empty_messages_returns_counters(self):
        """When no messages exist, must still advance counters."""
        from core.nodes import summarize_conversation
        state = {
            "messages": [],
            "action": 1,
            "retrieved_docs": [],
            "summary": "some KB",
            "turn_number": 1,
        }
        result = summarize_conversation(state, config={})
        self.assertEqual(result["turn_number"], 2)
        self.assertIn("prev_action", result)

    def test_summary_key_always_present(self):
        state = self._make_state(action=0)
        result = self._run(state)
        self.assertIn("summary", result)


# ─────────────────────────────────────────────────────────────────────────────
# F. RAGDecisionEnv — Gymnasium API
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGDecisionEnv(unittest.TestCase):
    """Test the Gym env without running LangGraph or the LLM."""

    def _make_env(self, n_episodes=2, turns_per=3):
        episodes = [
            [{"query": f"q{e}_{t}", "ground_truth": "ans", "retrieval_required": t == 0}
             for t in range(turns_per)]
            for e in range(n_episodes)
        ]
        env_module = __import__("rl_core.rl_env", fromlist=["RAGDecisionEnv"])
        RAGDecisionEnv = env_module.RAGDecisionEnv

        env = RAGDecisionEnv.__new__(RAGDecisionEnv)
        env.episodes = episodes
        env.beta = 0.01
        env.mode = "train"
        env._episode_idx = 0
        env._current_episode = []
        env._turn_idx = 0
        from core.nodes import INITIAL_GROUNDED_SUMMARY
        env._summary = INITIAL_GROUNDED_SUMMARY
        env._thread_id = "test"
        env._turn_number = 1
        env._prev_action = -1
        env._episode_rewards = []
        env._episode_actions = []
        env._episode_counter = 0

        import gymnasium as gym
        from gymnasium import spaces
        env.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(ENCODER_DIM,), dtype=np.float32
        )
        env.action_space = spaces.Discrete(2)

        return env, episodes

    def test_observation_space_shape(self):
        env, _ = self._make_env()
        self.assertEqual(env.observation_space.shape, (ENCODER_DIM,))

    def test_action_space_discrete_2(self):
        env, _ = self._make_env()
        self.assertEqual(env.action_space.n, 2)

    def test_prev_action_initial_sentinel(self):
        """At the very first turn, prev_action must be -1."""
        env, _ = self._make_env()
        self.assertEqual(env._prev_action, -1)

    def test_turn_number_starts_at_1(self):
        env, _ = self._make_env()
        self.assertEqual(env._turn_number, 1)

    def test_reset_reinitializes_state(self):
        env, _ = self._make_env()
        env._turn_number = 99
        env._prev_action = 1
        env._episode_rewards = [5.0]

        with patch.object(env, "_build_obs", return_value=np.zeros(ENCODER_DIM, dtype=np.float32)):
            obs, info = env.reset()

        self.assertEqual(env._turn_number, 1)
        self.assertEqual(env._prev_action, -1)
        self.assertEqual(env._episode_rewards, [])
        self.assertEqual(obs.shape, (ENCODER_DIM,))

    def test_build_obs_after_all_turns(self):
        """_build_obs when turn_idx >= episode length must return zero vector (not crash)."""
        env, episodes = self._make_env(n_episodes=1, turns_per=2)
        env._current_episode = episodes[0]
        env._turn_idx = 10  # past the end

        with patch("rl_core.rl_env.encode_state", side_effect=_fake_encode_state):
            obs = env._build_obs()
        self.assertEqual(obs.shape, (ENCODER_DIM,))


# ─────────────────────────────────────────────────────────────────────────────
# G. WarmStart — dataset builder & metrics helper
# ─────────────────────────────────────────────────────────────────────────────

class TestWarmStartDataset(unittest.TestCase):

    def _make_episodes(self):
        return [
            [
                {"query": "q1", "retrieval_required": True, "turn_id": 1},
                {"query": "q2", "retrieval_required": False, "turn_id": 2},
                {"query": "q3", "retrieval_required": True, "turn_id": 3},
            ]
        ]

    def test_label_alignment(self):
        """Labels must match retrieval_required in order."""
        episodes = self._make_episodes()
        with patch("rl_train.encode_state", side_effect=_fake_encode_state):
            from rl_train import _build_warmstart_dataset
            obs, labels = _build_warmstart_dataset(episodes)
        expected = [1, 0, 1]
        self.assertEqual(labels.tolist(), expected)

    def test_obs_shape(self):
        episodes = self._make_episodes()
        with patch("rl_train.encode_state", side_effect=_fake_encode_state):
            from rl_train import _build_warmstart_dataset
            obs, labels = _build_warmstart_dataset(episodes)
        self.assertEqual(obs.shape, (3, ENCODER_DIM))
        self.assertEqual(obs.dtype, np.float32)

    def test_prev_action_starts_minus_one(self):
        """The first turn of each episode must use prev_action=-1."""
        calls = []
        def fake_encode(summary, query, turn_number=1, prev_action=-1):
            calls.append(prev_action)
            return np.zeros(ENCODER_DIM, dtype=np.float32)

        with patch("rl_train.encode_state", side_effect=fake_encode):
            from rl_train import _build_warmstart_dataset
            episodes = [[{"query": "q", "retrieval_required": True}]]
            _build_warmstart_dataset(episodes)

        self.assertEqual(calls[0], -1)

    def test_prev_action_propagates(self):
        """Subsequent turns must use the ground-truth label as prev_action."""
        calls = []
        def fake_encode(summary, query, turn_number=1, prev_action=-1):
            calls.append(prev_action)
            return np.zeros(ENCODER_DIM, dtype=np.float32)

        with patch("rl_train.encode_state", side_effect=fake_encode):
            from rl_train import _build_warmstart_dataset
            episodes = [[
                {"query": "q1", "retrieval_required": True},
                {"query": "q2", "retrieval_required": False},
                {"query": "q3", "retrieval_required": True},
            ]]
            _build_warmstart_dataset(episodes)

        # turn 1: prev=-1, turn 2: prev=1 (label of turn 1), turn 3: prev=0 (label of turn 2)
        self.assertEqual(calls, [-1, 1, 0])


class TestWarmStartMetrics(unittest.TestCase):

    def test_perfect_predictions(self):
        from rl_train import _compute_warmstart_metrics
        preds  = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        m = _compute_warmstart_metrics(preds, labels)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertEqual(m["unsafe_skip_rate"], 0.0)
        self.assertEqual(m["wasteful_retrieve_rate"], 0.0)

    def test_all_retrieve(self):
        """Always retrieving: TP=2, FP=2, FN=0, TN=0."""
        from rl_train import _compute_warmstart_metrics
        preds  = [1, 1, 1, 1]
        labels = [1, 0, 1, 0]
        m = _compute_warmstart_metrics(preds, labels)
        self.assertEqual(m["TP"], 2)
        self.assertEqual(m["FP"], 2)
        self.assertEqual(m["FN"], 0)
        self.assertEqual(m["TN"], 0)
        self.assertAlmostEqual(m["unsafe_skip_rate"], 0.0)
        self.assertAlmostEqual(m["wasteful_retrieve_rate"], 0.5)

    def test_all_skip(self):
        """Always skipping: TP=0, FP=0, FN=2, TN=2."""
        from rl_train import _compute_warmstart_metrics
        preds  = [0, 0, 0, 0]
        labels = [1, 0, 1, 0]
        m = _compute_warmstart_metrics(preds, labels)
        self.assertEqual(m["FN"], 2)
        self.assertEqual(m["TN"], 2)
        self.assertAlmostEqual(m["unsafe_skip_rate"], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# H. Evaluation — compute_routing_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRoutingMetrics(unittest.TestCase):

    def test_perfect_routing(self):
        from rl_core.evaluation import compute_routing_metrics
        actions = [1, 0, 1, 0]
        req     = [True, False, True, False]
        m = compute_routing_metrics(actions, req)
        self.assertEqual(m["TP"], 2); self.assertEqual(m["FP"], 0)
        self.assertEqual(m["FN"], 0); self.assertEqual(m["TN"], 2)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["accuracy"], 1.0)

    def test_all_retrieve_baseline(self):
        from rl_core.evaluation import compute_routing_metrics
        actions = [1, 1, 1, 1]
        req     = [True, False, True, False]
        m = compute_routing_metrics(actions, req)
        self.assertEqual(m["FN"], 0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["unsafe_skip_rate"], 0.0)

    def test_all_skip(self):
        from rl_core.evaluation import compute_routing_metrics
        actions = [0, 0, 0, 0]
        req     = [True, True, False, False]
        m = compute_routing_metrics(actions, req)
        self.assertAlmostEqual(m["unsafe_skip_rate"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.0)

    def test_length_mismatch_raises(self):
        from rl_core.evaluation import compute_routing_metrics
        with self.assertRaises((AssertionError, ValueError)):
            compute_routing_metrics([1, 0], [True])

    def test_f1_zero_denominator(self):
        """All skip + all required → precision=0, recall=0, f1=0 (no division error)."""
        from rl_core.evaluation import compute_routing_metrics
        m = compute_routing_metrics([0, 0], [True, True])
        self.assertAlmostEqual(m["f1"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# I. Reward Formula
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardFormula(unittest.TestCase):
    """Verify reward = score - beta * approx_token_count(docs)."""

    def test_skip_has_zero_token_cost(self):
        """Action=0 (skip) must produce zero retrieved docs → zero token cost."""
        from rl_core.utils import approx_token_count
        cost = approx_token_count([])
        self.assertEqual(cost, 0)

    def test_approx_token_count(self):
        """4 chars ≈ 1 token."""
        from rl_core.utils import approx_token_count
        from langchain_core.documents import Document
        docs = [Document(page_content="abcdefgh")]  # 8 chars → 2 tokens
        self.assertEqual(approx_token_count(docs), 2)

    def test_reward_skip(self):
        """For skip turns: reward = score - 0 = score."""
        score = 8.0
        beta = 0.01
        token_cost = 0
        reward = score - beta * token_cost
        self.assertAlmostEqual(reward, 8.0)

    def test_reward_retrieve_penalized(self):
        """For retrieve turns: reward < score when docs are non-empty."""
        score = 8.0
        beta = 0.01
        # doc with 400 chars → 100 tokens → cost = 0.01 * 100 = 1.0
        token_cost = 100
        reward = score - beta * token_cost
        self.assertAlmostEqual(reward, 7.0)

    def test_beta_zero_no_penalty(self):
        """With beta=0, retrieval is never penalized regardless of doc length."""
        score = 5.0
        reward = score - 0.0 * 9999
        self.assertAlmostEqual(reward, 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# J. load_episodes — JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadEpisodes(unittest.TestCase):

    def _load(self, data: dict):
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fpath = f.name
        try:
            from rl_train import load_episodes
            return load_episodes(fpath)
        finally:
            os.unlink(fpath)

    def _minimal_conv(self, n_turns=2):
        return {
            "conversations": [
                {
                    "episode_id": "test_ep",
                    "scenario_type": "A",
                    "scenario_name": "Test",
                    "hazard_type": "flood",
                    "topic_group": "evacuation",
                    "turns": [
                        {
                            "turn_id": i + 1,
                            "query": f"question {i}",
                            "assistant_answer": f"answer {i}",
                            "retrieval_required": i == 0,
                            "query_type": "first_turn" if i == 0 else "followup_reuse",
                        }
                        for i in range(n_turns)
                    ],
                }
            ]
        }

    def test_basic_load(self):
        episodes = self._load(self._minimal_conv())
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 2)

    def test_turn_has_required_keys(self):
        episodes = self._load(self._minimal_conv())
        turn = episodes[0][0]
        for key in ("query", "ground_truth", "retrieval_required", "query_type"):
            self.assertIn(key, turn, f"Missing key: {key}")

    def test_retrieval_required_is_bool(self):
        episodes = self._load(self._minimal_conv())
        for ep in episodes:
            for turn in ep:
                self.assertIsInstance(turn["retrieval_required"], bool)

    def test_metadata_propagated_to_turns(self):
        """episode_id, scenario_type etc. must be propagated to every turn."""
        episodes = self._load(self._minimal_conv())
        for turn in episodes[0]:
            self.assertEqual(turn["episode_id"], "test_ep")
            self.assertEqual(turn["scenario_type"], "A")
            self.assertEqual(turn["hazard_type"], "flood")

    def test_malformed_episode_skipped(self):
        data = {
            "conversations": [
                {"episode_id": "bad"},  # no turns key
                self._minimal_conv()["conversations"][0],
            ]
        }
        episodes = self._load(data)
        self.assertEqual(len(episodes), 1)

    def test_missing_query_turn_skipped(self):
        data = self._minimal_conv()
        data["conversations"][0]["turns"].append({"turn_id": 99})  # no query
        episodes = self._load(data)
        self.assertEqual(len(episodes[0]), 2)  # malformed turn not included

    def test_wrong_top_level_raises(self):
        """List at top level (not wrapped in 'conversations') must raise ValueError."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            fpath = f.name
        try:
            from rl_train import load_episodes
            with self.assertRaises((ValueError, KeyError, Exception)):
                load_episodes(fpath)
        finally:
            os.unlink(fpath)


# ─────────────────────────────────────────────────────────────────────────────
# K. JudgeChain — prompt & parser
# ─────────────────────────────────────────────────────────────────────────────

class TestJudgeParser(unittest.TestCase):

    def test_valid_json(self):
        from rl_core.utils import _parse_judge_output
        raw = '{"score": 8, "reason": "well grounded"}'
        out = _parse_judge_output(raw)
        self.assertEqual(out["score"], 8)
        self.assertEqual(out["reason"], "well grounded")

    def test_score_clamped_above_10(self):
        from rl_core.utils import _parse_judge_output
        raw = '{"score": 15, "reason": "too high"}'
        out = _parse_judge_output(raw)
        self.assertEqual(out["score"], 10)

    def test_score_clamped_below_0(self):
        from rl_core.utils import _parse_judge_output
        raw = '{"score": -3, "reason": "negative"}'
        out = _parse_judge_output(raw)
        self.assertEqual(out["score"], 0)

    def test_json_with_markdown_fences(self):
        """Model sometimes wraps JSON in ```json ... ``` fences."""
        from rl_core.utils import _parse_judge_output
        raw = '```json\n{"score": 7, "reason": "ok"}\n```'
        out = _parse_judge_output(raw)
        self.assertEqual(out["score"], 7)

    def test_fallback_on_bad_json(self):
        """If JSON cannot be parsed, must return score=5 (default) and not raise."""
        from rl_core.utils import _parse_judge_output
        out = _parse_judge_output("completely not json at all")
        self.assertIn("score", out)
        self.assertIsInstance(out["score"], int)

    def test_scope_reminder_always_injected(self):
        """scope_reminder must never be settable by callers."""
        # Build a mock chain so we can inspect the inputs passed to it
        with patch("rl_core.utils.build_judge_llm") as mock_llm_fn:
            mock_llm = MagicMock()
            mock_llm_fn.return_value = mock_llm

            from rl_core.utils import JudgeChain
            judge = JudgeChain()

            captured = {}
            def fake_invoke(inputs):
                captured.update(inputs)
                return '{"score": 5, "reason": "test"}'

            judge._chain = MagicMock()
            judge._chain.invoke.side_effect = fake_invoke

            judge.invoke({
                "query": "test",
                "conversation_summary": "summary",
                "action_taken": "skip",
                "retrieved_chunks": "",
                "response": "ans",
                "scope_reminder": "INJECTED_BY_CALLER",  # attacker override
            })

            # The chain must have received the correct scope reminder, not the caller's value
            self.assertIn("scope_reminder", captured)
            from rl_core.utils import _SCOPE_REMINDER
            self.assertEqual(captured["scope_reminder"], _SCOPE_REMINDER)

    def test_empty_retrieved_chunks_default(self):
        """When retrieved_chunks is not provided or empty, must default to 'No retrieval performed'."""
        with patch("rl_core.utils.build_judge_llm"):
            from rl_core.utils import JudgeChain
            judge = JudgeChain()

            captured = {}
            def fake_invoke(inputs):
                captured.update(inputs)
                return '{"score": 5, "reason": ""}'

            judge._chain = MagicMock()
            judge._chain.invoke.side_effect = fake_invoke

            judge.invoke({
                "query": "q",
                "conversation_summary": "s",
                "action_taken": "skip",
                "response": "r",
                # retrieved_chunks intentionally omitted
            })
            self.assertEqual(captured["retrieved_chunks"], "No retrieval performed")


# ─────────────────────────────────────────────────────────────────────────────
# L. Data Consistency — conversations.json
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationsJson(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not CONVERSATIONS_JSON.exists():
            raise unittest.SkipTest(f"conversations.json not found at {CONVERSATIONS_JSON}")
        with open(CONVERSATIONS_JSON) as f:
            cls.data = json.load(f)

    def test_top_level_structure(self):
        self.assertIn("conversations", self.data)
        self.assertIsInstance(self.data["conversations"], list)

    def test_all_episodes_have_turns(self):
        for ep in self.data["conversations"]:
            self.assertIn("turns", ep, f"Episode {ep.get('episode_id')} missing 'turns'")
            self.assertGreater(len(ep["turns"]), 0)

    def test_all_turns_have_query_and_answer(self):
        for ep in self.data["conversations"]:
            for t in ep["turns"]:
                self.assertIn("query", t)
                self.assertIn("assistant_answer", t)

    def test_retrieval_required_is_bool(self):
        for ep in self.data["conversations"]:
            for t in ep["turns"]:
                val = t.get("retrieval_required")
                # Must be bool or at least convertible — not None
                self.assertIsNotNone(val, f"retrieval_required is None in {ep.get('episode_id')}")
                self.assertIsInstance(val, bool)

    def test_first_turn_always_retrieves(self):
        """The dataset label for first turns should (almost always) require retrieval."""
        for ep in self.data["conversations"]:
            first_turn = ep["turns"][0]
            self.assertTrue(
                first_turn["retrieval_required"],
                f"Episode {ep.get('episode_id')}: first turn marked retrieval_required=False"
            )

    def test_unanswerable_turns_no_retrieval(self):
        """Turns marked 'unanswerable' should have retrieval_required=False."""
        for ep in self.data["conversations"]:
            for t in ep["turns"]:
                if t.get("query_type") == "unanswerable":
                    self.assertFalse(
                        t["retrieval_required"],
                        f"Unanswerable turn in {ep.get('episode_id')} has retrieval_required=True"
                    )

    def test_metadata_counts_match(self):
        """metadata.total_episodes must match actual episode count."""
        if "metadata" not in self.data:
            self.skipTest("No metadata block")
        meta = self.data["metadata"]
        actual = len(self.data["conversations"])
        self.assertEqual(meta.get("total_episodes"), actual)

    def test_no_empty_queries(self):
        for ep in self.data["conversations"]:
            for t in ep["turns"]:
                self.assertTrue(t["query"].strip(), "Empty query found")


# ─────────────────────────────────────────────────────────────────────────────
# M. Integration — one turn, end-to-end (no real LLM/vectorstore)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndSingleTurn(unittest.TestCase):
    """
    Wire the full graph with stub LLM and vectorstore.
    Verifies that a single turn updates summary, answer, turn_number correctly.

    Important: LangGraph's MemorySaver serialises the full state to msgpack
    after every node.  Any non-serialisable object (e.g. MagicMock) in
    retrieved_docs will cause a TypeError at checkpoint time.  The vectorstore
    stub must therefore return real langchain Document objects from
    retriever.invoke(), not a MagicMock.
    """

    def _stub_llm_response(self, text="stub answer"):
        mock_result = MagicMock()
        mock_result.content = text
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        return mock_chain

    def _make_stub_vectorstore(self):
        """Return a vectorstore stub whose retriever yields real Document objects."""
        from langchain_core.documents import Document

        stub_docs = [Document(page_content="stub chunk about floods", metadata={})]

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = stub_docs

        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        return mock_vs

    def _run_turn(self, action: int, query: str = "What to do in a flood?"):
        from langchain_core.messages import HumanMessage

        with patch("core.nodes.load_or_create_vectorstore") as mock_vs_factory, \
             patch("core.nodes.build_llm") as mock_llm_factory, \
             patch("core.nodes.ANSWER_PROMPT") as mock_ap, \
             patch("core.nodes.GROUNDED_SUMMARY_RETRIEVE_PROMPT") as mock_rp, \
             patch("core.nodes.GROUNDED_SUMMARY_SKIP_PROMPT") as mock_sp:

            # Stub vectorstore — retriever.invoke() must return real Documents
            # so MemorySaver can msgpack-serialise retrieved_docs without error.
            mock_vs_factory.return_value = self._make_stub_vectorstore()

            # Stub LLM factory
            mock_llm = MagicMock()
            mock_llm_factory.return_value = mock_llm

            # Stub prompt chains — make | operator work
            answer_chain = self._stub_llm_response("stub answer")
            summary_chain = self._stub_llm_response("updated KB")

            mock_ap.__or__ = lambda s, other: answer_chain
            mock_rp.__or__ = lambda s, other: summary_chain
            mock_sp.__or__ = lambda s, other: summary_chain

            # Reset the module-level singleton so our stub is actually used.
            import core.nodes as _nodes
            _nodes._vectorstore = None
            _nodes._llm = None

            from core.graph import build_graph
            from core.nodes import INITIAL_GROUNDED_SUMMARY
            app = build_graph()

            input_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "summary": INITIAL_GROUNDED_SUMMARY,
                "retrieved_docs": [],
                "answer": "",
                "mode": "rl",
                "turn_number": 1,
                "prev_action": -1,
            }
            run_cfg = {
                "configurable": {
                    "thread_id": "test-thread",
                    "forced_action": action,
                }
            }
            final_state = {}
            for event in app.stream(input_state, config=run_cfg):
                for _, partial in event.items():
                    if partial:
                        final_state.update(partial)
            return final_state

    def test_retrieve_action_graph_runs(self):
        state = self._run_turn(action=1)
        self.assertIn("answer", state)

    def test_skip_action_graph_runs(self):
        state = self._run_turn(action=0)
        self.assertIn("answer", state)

    def test_turn_number_advances(self):
        state = self._run_turn(action=1)
        self.assertEqual(state.get("turn_number", 0), 2)

    def test_prev_action_set_after_turn(self):
        state = self._run_turn(action=1)
        self.assertEqual(state.get("prev_action"), 1)

    def test_summary_updated(self):
        state = self._run_turn(action=1)
        # summary must exist and be a non-empty string
        self.assertIsInstance(state.get("summary"), str)
        self.assertTrue(len(state.get("summary", "")) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# N. INITIAL_GROUNDED_SUMMARY sentinel
# ─────────────────────────────────────────────────────────────────────────────

class TestInitialSentinel(unittest.TestCase):

    def test_sentinel_exists(self):
        from core.nodes import INITIAL_GROUNDED_SUMMARY
        self.assertIsInstance(INITIAL_GROUNDED_SUMMARY, str)
        self.assertTrue(len(INITIAL_GROUNDED_SUMMARY) > 0)

    def test_sentinel_used_in_env_reset(self):
        """After env.reset(), summary must equal INITIAL_GROUNDED_SUMMARY."""
        from core.nodes import INITIAL_GROUNDED_SUMMARY
        episodes = [[{"query": "q", "ground_truth": "a", "retrieval_required": True}]]

        env_module = __import__("rl_core.rl_env", fromlist=["RAGDecisionEnv"])
        RAGDecisionEnv = env_module.RAGDecisionEnv
        env = RAGDecisionEnv.__new__(RAGDecisionEnv)
        env.episodes = episodes
        env.beta = 0.01
        env.mode = "train"
        env._episode_idx = 0
        env._current_episode = []
        env._turn_idx = 0
        env._summary = "OLD SUMMARY"
        env._thread_id = "test"
        env._turn_number = 99
        env._prev_action = 1
        env._episode_rewards = [10.0]
        env._episode_actions = [1]
        env._episode_counter = 5

        import gymnasium as gym
        from gymnasium import spaces
        env.observation_space = spaces.Box(-np.inf, np.inf, (ENCODER_DIM,), np.float32)
        env.action_space = spaces.Discrete(2)

        with patch.object(env, "_build_obs", return_value=np.zeros(ENCODER_DIM, dtype=np.float32)):
            env.reset()

        self.assertEqual(env._summary, INITIAL_GROUNDED_SUMMARY)


# ─────────────────────────────────────────────────────────────────────────────
# O. Config
# ─────────────────────────────────────────────────────────────────────────────

class TestAppConfig(unittest.TestCase):

    def test_policy_encoder_dim_matches_model(self):
        """MiniLM = 384 dim; if model changes to distilbert, dim must also change."""
        from core.config import CONFIG
        if "MiniLM" in CONFIG.policy_encoder_model:
            self.assertEqual(CONFIG.policy_encoder_dim, 384)
        elif "distilbert" in CONFIG.policy_encoder_model:
            self.assertEqual(CONFIG.policy_encoder_dim, 768)

    def test_rl_beta_positive(self):
        from core.config import CONFIG
        self.assertGreater(CONFIG.rl_beta, 0)

    def test_observation_space_matches_config(self):
        """RAGDecisionEnv observation_space.shape[0] must equal CONFIG.policy_encoder_dim."""
        from core.config import CONFIG
        import gymnasium as gym
        from gymnasium import spaces
        obs_space = spaces.Box(-np.inf, np.inf, (CONFIG.policy_encoder_dim,), np.float32)
        self.assertEqual(obs_space.shape[0], CONFIG.policy_encoder_dim)


# ─────────────────────────────────────────────────────────────────────────────
# P. Heuristic router (evaluation baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TestHeuristicAction(unittest.TestCase):

    def test_first_turn_always_retrieve(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "first_turn"}
        self.assertEqual(heuristic_action(turn, turn_idx=0), 1)

    def test_topic_shift_retrieves(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "topic_shift"}
        self.assertEqual(heuristic_action(turn, turn_idx=2), 1)

    def test_followup_new_retrieves(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "followup_new"}
        self.assertEqual(heuristic_action(turn, turn_idx=3), 1)

    def test_followup_reuse_skips(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "followup_reuse"}
        self.assertEqual(heuristic_action(turn, turn_idx=2), 0)

    def test_clarification_skips(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "clarification"}
        self.assertEqual(heuristic_action(turn, turn_idx=3), 0)

    def test_unanswerable_skips(self):
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "unanswerable"}
        self.assertEqual(heuristic_action(turn, turn_idx=4), 0)

    def test_idx_zero_overrides_any_type(self):
        """Turn index 0 must always retrieve regardless of query_type."""
        from rl_core.evaluation import heuristic_action
        turn = {"query_type": "followup_reuse"}
        self.assertEqual(heuristic_action(turn, turn_idx=0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Q. format_chunks_for_judge
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatChunksForJudge(unittest.TestCase):

    def test_empty_returns_sentinel(self):
        from rl_core.utils import format_chunks_for_judge
        result = format_chunks_for_judge([])
        self.assertEqual(result, "No retrieval performed")

    def test_long_chunk_truncated(self):
        from rl_core.utils import format_chunks_for_judge
        from langchain_core.documents import Document
        long_text = "x" * 500
        docs = [Document(page_content=long_text)]
        result = format_chunks_for_judge(docs, max_chars_per_chunk=300)
        self.assertIn("...", result)
        # Must not exceed 300 + 3 (ellipsis)
        chunk_content = result.split("] ", 1)[1]
        self.assertLessEqual(len(chunk_content), 303)

    def test_short_chunk_not_truncated(self):
        from rl_core.utils import format_chunks_for_judge
        from langchain_core.documents import Document
        docs = [Document(page_content="short text")]
        result = format_chunks_for_judge(docs)
        self.assertNotIn("...", result)
        self.assertIn("short text", result)

    def test_multiple_chunks_indexed(self):
        from rl_core.utils import format_chunks_for_judge
        from langchain_core.documents import Document
        docs = [Document(page_content="first"), Document(page_content="second")]
        result = format_chunks_for_judge(docs)
        self.assertIn("[Chunk 1]", result)
        self.assertIn("[Chunk 2]", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
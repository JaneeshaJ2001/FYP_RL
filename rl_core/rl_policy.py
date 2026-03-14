"""
rl_policy.py
────────────
Custom Stable-Baselines3 ActorCriticPolicy.

Architecture
────────────
  Frozen transformer backbone  (encode_state from state_encoder.py)
       │
       ▼  (encoder_dim = 384 or 768)
  Shared MLP trunk  Linear(dim→256) → ReLU → Linear(256→64) → ReLU
       │
       ├─► Action head  Linear(64 → n_actions)   [actor]
       └─► Value head   Linear(64 → 1)            [critic]

ONLY the MLP layers are registered as trainable parameters.
The transformer is NEVER updated.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from core.config import CONFIG

logger = logging.getLogger("disaster_chatbot")


# ──────────────────────────────────────────────────────────────────────────────
# Feature extractor: wraps the frozen encoder inside SB3's extractor API
# ──────────────────────────────────────────────────────────────────────────────

class FrozenTransformerExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor that delegates to the shared frozen transformer.

    The observation is already a float32 numpy vector (produced by encode_state).
    This extractor simply passes it through as-is — the transformer ran earlier
    in the environment step.  This avoids double-inference.

    If you ever want to run the transformer INSIDE the policy (e.g., for GPU
    batching), swap encode_state for encode_tensor here.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        features_dim = int(np.prod(observation_space.shape))
        super().__init__(observation_space, features_dim=features_dim)
        # No parameters here — observation IS the embedding already.

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, encoder_dim)
        return observations


# ──────────────────────────────────────────────────────────────────────────────
# Custom ActorCriticPolicy
# ──────────────────────────────────────────────────────────────────────────────

class TransformerHeadPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy with:
      • Frozen transformer backbone (handled externally via state_encoder)
      • Trainable shared MLP trunk
      • Trainable action head (actor)
      • Trainable value head (critic)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        *args,
        **kwargs,
    ):
        # Disable SB3's default MLP builder
        kwargs["net_arch"] = []
        kwargs["features_extractor_class"] = FrozenTransformerExtractor
        kwargs["features_extractor_kwargs"] = {}

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        # NOTE: mlp_trunk / action_head / value_head are created in
        # _build_mlp_extractor(), which SB3 calls inside _build() *before*
        # the optimizer is constructed from self.parameters().  Placing them
        # here (after super().__init__) would mean they are absent from the
        # optimizer because the optimizer is already built by the time
        # super().__init__() returns.

    # ── SB3 hooks ─────────────────────────────────────────────────────────────

    def _build_mlp_extractor(self) -> None:
        """Build all trainable layers here so they exist BEFORE SB3's _build()
        creates the optimizer from self.parameters().

        A dummy mlp_extractor with empty arch is created so that SB3's internal
        helpers (e.g. set_training_mode) can access the attribute without error.
        """
        from stable_baselines3.common.torch_layers import MlpExtractor
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=[],
            activation_fn=nn.ReLU,
            device=self.device,
        )

        encoder_dim = self.features_dim   # == observation_space.shape[0], always in sync
        n_actions = self.action_space.n  # type: ignore[attr-defined]

        # ── Shared MLP trunk ─────────────────────────────────────────────────
        self.mlp_trunk = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # ── Actor head ───────────────────────────────────────────────────────
        self.action_head = nn.Linear(64, n_actions)

        # ── Critic (value) head ──────────────────────────────────────────────
        self.value_head = nn.Linear(64, 1)

        # Verify: only MLP parameters should be trainable
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TransformerHeadPolicy: %d trainable parameters (MLP heads only)",
            trainable,
        )

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        trunk_out = self.mlp_trunk(features)
        logits = self.action_head(trunk_out)
        values = self.value_head(trunk_out).squeeze(-1)
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor):  # type: ignore
        return self.action_dist.proba_distribution(action_logits=latent_pi)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)
        trunk_out = self.mlp_trunk(features)
        logits = self.action_head(trunk_out)
        values = self.value_head(trunk_out).squeeze(-1)
        distribution = self._get_action_dist_from_latent(logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        """Actor: obs → trunk → action_head → categorical distribution."""
        features = self.extract_features(obs)
        trunk_out = self.mlp_trunk(features)
        logits = self.action_head(trunk_out)
        return self._get_action_dist_from_latent(logits)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Critic: obs → trunk → value_head → scalar value estimate."""
        features = self.extract_features(obs)
        trunk_out = self.mlp_trunk(features)
        return self.value_head(trunk_out).squeeze(-1)
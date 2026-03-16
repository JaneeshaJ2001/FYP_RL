"""
state_encoder.py
────────────────
Frozen transformer encoder used by:
  • The custom SB3 ActorCriticPolicy (feature extractor)
  • The Gym environment (observation builder)

The transformer weights are NEVER updated during PPO training.
Only the MLP head (defined in rl_policy.py) is trainable.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from core.config import CONFIG

logger = logging.getLogger("disaster_chatbot")

# ── Singleton ──────────────────────────────────────────────────────────────────
_encoder_model: AutoModel | None = None
_encoder_tokenizer: AutoTokenizer | None = None
_encoder_device: str = "cpu"


def _get_encoder():
    global _encoder_model, _encoder_tokenizer, _encoder_device

    if _encoder_model is not None:
        return _encoder_model, _encoder_tokenizer, _encoder_device

    model_name = CONFIG.policy_encoder_model
    logger.info("Loading frozen encoder: %s", model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # ── FREEZE ALL TRANSFORMER WEIGHTS ────────────────────────────────────────
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # ──────────────────────────────────────────────────────────────────────────

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    _encoder_model = model
    _encoder_tokenizer = tokenizer
    _encoder_device = device

    logger.info("Encoder loaded and frozen on device=%s", device)
    return _encoder_model, _encoder_tokenizer, _encoder_device


def _build_encoder_text(
    summary: str,
    query: str,
    turn_number: int,
    prev_action: int,
) -> str:
    """
    Build the full text string fed to the frozen transformer.

    Format (all sections separated by [SEP]):
        [TURN <N>] [SEP] [PREV_ACTION <label>] [SEP] <summary> [SEP] <query>

    Where:
        <N>      = 1-based turn number within the episode
        <label>  = "retrieve" | "skip" | "none" (first turn)

    Keeping the prefix short ensures the query and summary get the most tokens
    within the 256-token budget.
    """
    # Map numeric prev_action to a human-readable token the encoder can use
    if prev_action == 1:
        prev_label = "retrieve"
    elif prev_action == 0:
        prev_label = "skip"
    else:
        prev_label = "none"          # -1 sentinel: first turn of episode

    prefix = f"[TURN {turn_number}] [SEP] [PREV_ACTION {prev_label}]"

    if summary.strip():
        return f"{prefix} [SEP] {summary} [SEP] {query}"
    else:
        return f"{prefix} [SEP] {query}"


def encode_state(
    summary: str,
    query: str,
    turn_number: int = 1,     
    prev_action: int = -1,    
) -> np.ndarray:
    """
    Encode (summary, query, turn_number, prev_action) into a fixed-size float32
    numpy vector.

    Input format:
        "[TURN <N>] [SEP] [PREV_ACTION <label>] [SEP] {summary} [SEP] {query}"

    Output: mean-pooled last hidden state → shape (encoder_dim,)

    This function is called in two contexts:
      1. Inside RAGDecisionEnv.reset() / .step() to build observations.
      2. Inside decide_retrieve (policy mode) to build obs for inference.
    """
    model, tokenizer, device = _get_encoder()

    text = _build_encoder_text(summary, query, turn_number, prev_action)

    with torch.no_grad():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state   # (1, seq_len, hidden_dim)
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        sum_hidden = (last_hidden * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1e-9)
        embedding = (sum_hidden / count).squeeze(0)   # (hidden_dim,)

    return embedding.cpu().numpy().astype(np.float32)


def encode_tensor(
    summary: str,
    query: str,
    turn_number: int = 1,   
    prev_action: int = -1,  
) -> torch.Tensor:
    """Same as encode_state but returns a torch.Tensor (used inside policy net)."""
    model, tokenizer, device = _get_encoder()

    text = _build_encoder_text(summary, query, turn_number, prev_action)

    # NOTE: no torch.no_grad() here — caller controls grad context
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    sum_hidden = (last_hidden * attention_mask).sum(dim=1)
    count = attention_mask.sum(dim=1).clamp(min=1e-9)
    embedding = (sum_hidden / count).squeeze(0)

    return embedding
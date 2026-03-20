"""
rl_core/embed_cache.py
──────────────────────
Disk cache for warm-start observation embeddings.

With 3,500 conversations (~16,800 turns) encoding takes ~35s on CPU every run.
This module caches the (obs, label) arrays to disk as .npz files keyed by a
hash of the episode content, so repeated runs (ablations, debugging, resumed
training) skip re-encoding entirely.

Usage in rl_train.py:
    from rl_core.embed_cache import build_warmstart_dataset_cached

    train_obs, train_labels = build_warmstart_dataset_cached(train_episodes, split="train")
    val_obs,   val_labels   = build_warmstart_dataset_cached(val_episodes,   split="val")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("disaster_chatbot")

# Default cache directory — gitignored alongside chroma_db and policy_checkpoints
_DEFAULT_CACHE_DIR = Path(".embed_cache")


def _episode_hash(episodes: list[list[dict]]) -> str:
    """
    Stable SHA-256 hash of the episode content.

    Hashes only the fields that affect the encoded observation:
        query, retrieval_required (label), turn order.
    Ignores metadata fields that don't affect encoding (episode_id, etc.).
    """
    fingerprint = json.dumps(
        [
            [
                {
                    "q": t["query"],
                    "r": t.get("retrieval_required", True),
                }
                for t in ep
            ]
            for ep in episodes
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def build_warmstart_dataset_cached(
    episodes: list[list[dict]],
    split: str = "train",
    cache_dir: str | Path = _DEFAULT_CACHE_DIR,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (obs_array, label_array) for the given episodes.

    If a matching cache file exists (same content hash), load it instantly.
    Otherwise encode from scratch, save to cache, then return.

    Parameters
    ----------
    episodes     : list of episode turn lists (from load_episodes)
    split        : "train" or "val" — used only in the cache filename
    cache_dir    : directory to store .npz cache files (default: .embed_cache/)
    force_rebuild: if True, ignore existing cache and re-encode

    Returns
    -------
    obs_array    : float32 ndarray, shape (N, encoder_dim)
    label_array  : int64  ndarray, shape (N,)
    """
    # Late import to avoid circular dependency
    from rl_train import _build_warmstart_dataset

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    content_hash = _episode_hash(episodes)
    cache_file   = cache_dir / f"warmstart_{split}_{content_hash}.npz"

    if cache_file.exists() and not force_rebuild:
        logger.info("Loading cached embeddings from %s", cache_file)
        t0 = time.time()
        data = np.load(cache_file)
        obs, labels = data["obs"], data["labels"]
        logger.info(
            "Cache hit: loaded %d observations in %.1fs", len(labels), time.time() - t0
        )
        return obs, labels

    logger.info(
        "Cache miss (%s split, hash=%s) — encoding %d episodes …",
        split, content_hash, len(episodes),
    )
    t0 = time.time()
    obs, labels = _build_warmstart_dataset(episodes)
    elapsed = time.time() - t0
    logger.info(
        "Encoded %d observations in %.1fs — saving to %s",
        len(labels), elapsed, cache_file,
    )
    np.savez_compressed(cache_file, obs=obs, labels=labels)
    return obs, labels


def clear_cache(cache_dir: str | Path = _DEFAULT_CACHE_DIR) -> int:
    """Delete all .npz cache files. Returns count of files deleted."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    deleted = 0
    for f in cache_dir.glob("warmstart_*.npz"):
        f.unlink()
        deleted += 1
    logger.info("Cleared %d cache file(s) from %s", deleted, cache_dir)
    return deleted
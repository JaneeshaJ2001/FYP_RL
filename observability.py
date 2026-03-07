"""Langfuse tracing and runtime config helpers."""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from langchain_core.runnables import RunnableConfig

from config import CONFIG

logger = logging.getLogger("disaster_chatbot")

# Langfuse tracing (optional, lazy-imported)
get_langfuse_client = None
LangfuseCallbackHandler = None

_langfuse_handler: Any = None
_langfuse_initialized = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_langfuse_handler():
    """Create a singleton Langfuse callback handler when credentials exist."""
    global _langfuse_handler, _langfuse_initialized
    global get_langfuse_client, LangfuseCallbackHandler

    if _langfuse_initialized:
        return _langfuse_handler

    _langfuse_initialized = True

    tracing_enabled = CONFIG.langfuse_enabled and _env_flag(
        "LANGFUSE_TRACING_ENABLED", default=True
    )
    if not tracing_enabled:
        logger.info("Langfuse tracing disabled via configuration.")
        return None

    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        logger.info("Langfuse keys not found in environment; tracing disabled.")
        return None

    if get_langfuse_client is None or LangfuseCallbackHandler is None:
        try:
            langfuse_module = importlib.import_module("langfuse")
            langfuse_lc_module = importlib.import_module("langfuse.langchain")
            get_langfuse_client = getattr(langfuse_module, "get_client")
            LangfuseCallbackHandler = getattr(langfuse_lc_module, "CallbackHandler")
        except Exception:
            logger.warning("Langfuse SDK not installed; tracing disabled.")
            return None

    try:
        get_langfuse_client()
        _langfuse_handler = LangfuseCallbackHandler()
        logger.info("Langfuse tracing enabled.")
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse tracing: %s", exc)
        _langfuse_handler = None

    return _langfuse_handler


def extract_invoke_config(config: RunnableConfig | None) -> dict[str, Any]:
    """Forward callbacks/metadata from graph runtime config to LangChain calls."""
    if not config:
        return {}

    invoke_config: dict[str, Any] = {}
    callbacks = config.get("callbacks")
    metadata = config.get("metadata")

    if callbacks:
        invoke_config["callbacks"] = callbacks
    if metadata:
        invoke_config["metadata"] = metadata

    return invoke_config


def build_run_config(thread_id: str) -> dict[str, Any]:
    """Build LangGraph run config and attach Langfuse callbacks when available."""
    run_config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    handler = _get_langfuse_handler()

    if handler is not None:
        run_config["callbacks"] = [handler]
        run_config["metadata"] = {
            "langfuse_session_id": thread_id,
            "langfuse_tags": list(CONFIG.langfuse_tags),
        }

    return run_config


def flush_langfuse() -> None:
    """Flush telemetry so short-lived runs do not lose traces on shutdown."""
    if _langfuse_handler is None or get_langfuse_client is None:
        return

    try:
        get_langfuse_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush skipped: %s", exc)

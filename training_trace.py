from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

TRACE_LOGGER_NAME = "training_trace"

_trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
_trace_logger.propagate = False


def configure_training_trace(log_path: str) -> Path:
    path = Path(log_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)

    resolved_path = path.resolve()
    _trace_logger.setLevel(logging.INFO)

    for handler in list(_trace_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler_path = Path(handler.baseFilename).resolve()
            if handler_path == resolved_path:
                return resolved_path
            _trace_logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(resolved_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _trace_logger.addHandler(file_handler)
    return resolved_path


def _clean_text(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def trace_event(event: str, **fields: Any) -> None:
    if not _trace_logger.handlers:
        return

    parts = [event]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            rendered = f"{value:.3f}"
        else:
            rendered = _clean_text(value)
        parts.append(f"{key}={rendered}")

    _trace_logger.info(" | ".join(parts))
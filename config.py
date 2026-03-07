"""Centralized runtime configuration for the Disaster RAG chatbot."""

from __future__ import annotations

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()


def _to_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _to_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _to_tags(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    tags = tuple(tag.strip() for tag in raw.split(",") if tag.strip())
    return tags or default


@dataclass(frozen=True, slots=True)
class AppConfig:
    # Vector store
    chroma_dir: str = "./chroma_db_disaster"
    chroma_collection: str = "disaster_rag"
    top_k: int = 3

    # Ingestion
    chunks_path: str = "chunks.json"

    # Embeddings
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM
    llm_provider: str = "groq"  # groq | openai
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1

    # Memory
    summary_max_tokens: int = 300

    # Observability
    langfuse_enabled: bool = True
    langfuse_tags: tuple[str, ...] = field(
        default_factory=lambda: ("disaster-rag", "langgraph")
    )

    @classmethod
    def from_env(cls) -> "AppConfig":
        defaults = cls()
        return cls(
            chroma_dir=os.getenv("CHROMA_DIR", defaults.chroma_dir),
            chroma_collection=os.getenv("CHROMA_COLLECTION", defaults.chroma_collection),
            top_k=_to_int(os.getenv("TOP_K"), defaults.top_k),
            chunks_path=os.getenv("CHUNKS_PATH", defaults.chunks_path),
            embed_model=os.getenv("EMBED_MODEL", defaults.embed_model),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider),
            llm_model=os.getenv("LLM_MODEL", defaults.llm_model),
            llm_temperature=_to_float(
                os.getenv("LLM_TEMPERATURE"), defaults.llm_temperature
            ),
            summary_max_tokens=_to_int(
                os.getenv("SUMMARY_MAX_TOKENS"), defaults.summary_max_tokens
            ),
            langfuse_enabled=_to_bool(
                os.getenv("LANGFUSE_ENABLED"), defaults.langfuse_enabled
            ),
            langfuse_tags=_to_tags(
                os.getenv("LANGFUSE_TAGS"), defaults.langfuse_tags
            ),
        )


CONFIG = AppConfig.from_env()

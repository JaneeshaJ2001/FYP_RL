"""Data loading and vectorstore ingestion helpers."""

from __future__ import annotations

import ast
import json
import logging
import os

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import CONFIG

logger = logging.getLogger("disaster_chatbot")


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=CONFIG.embed_model)


def _load_chunks_file(file_path: str) -> "pd.DataFrame":
    """Load chunk data from supported JSON chunk schemas."""
    with open(file_path, "r", encoding="utf-8") as file_handle:
        raw = file_handle.read().strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot parse {file_path}.\n"
            "Expected valid JSON containing chunk records."
        ) from exc

    records: list[object] | None = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        # Support wrapped payloads such as {"sop_chunks": [...]}.
        for key in ("sop_chunks", "chunks", "records", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                records = candidate
                logger.info("Detected wrapped chunks payload under key '%s'", key)
                break

    if not isinstance(records, list):
        raise ValueError(
            f"Cannot parse {file_path}.\n"
            "Expected: JSON array [ {...}, ... ] or object containing a list under "
            "'sop_chunks'."
        )

    normalized_records: list[dict[str, object]] = []
    skipped_records = 0
    for record in records:
        if not isinstance(record, dict):
            skipped_records += 1
            continue

        chunk = record.get("chunk")
        if chunk is None:
            chunk = record.get("text")
        if chunk is None:
            chunk = record.get("content")

        if chunk is None:
            skipped_records += 1
            continue

        metadata = record.get("metadata")
        if metadata is None:
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"chunk", "text", "content", "metadata"}
            }

        normalized_records.append({"chunk": str(chunk), "metadata": metadata})

    if not normalized_records:
        raise ValueError(
            f"Cannot parse {file_path}.\n"
            "Expected records with a 'chunk' or 'text' field and optional 'metadata'."
        )

    df = pd.DataFrame(normalized_records)
    logger.info(
        "Loaded %d chunk records from %s (skipped=%d)",
        len(df),
        file_path,
        skipped_records,
    )
    return df


def load_or_create_vectorstore(
    json_path: str | None = None,
    force_recreate: bool = False,
) -> Chroma:
    """Load existing Chroma collection from disk, or create it from a JSON file."""
    embeddings = build_embeddings()
    chroma_dir = CONFIG.chroma_dir
    collection = CONFIG.chroma_collection

    db_exists = os.path.isdir(chroma_dir) and any(
        file_name.endswith(".sqlite3") for file_name in os.listdir(chroma_dir)
    ) if os.path.isdir(chroma_dir) else False

    if db_exists and not force_recreate:
        logger.info("Loading existing Chroma DB from %s", chroma_dir)
        vectorstore = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=chroma_dir,
        )
        logger.info(
            "Collection '%s' loaded (count=%s)",
            collection,
            vectorstore._collection.count(),
        )
        return vectorstore

    if json_path is None:
        json_path = CONFIG.chunks_path

    logger.info("Creating new Chroma DB from %s", json_path)
    df = _load_chunks_file(json_path)

    documents: list[Document] = []
    for _, row in df.iterrows():
        chunk = str(row["chunk"])

        raw_meta = row.get("metadata", "{}")
        if isinstance(raw_meta, str):
            try:
                meta = json.loads(raw_meta)
            except json.JSONDecodeError:
                try:
                    meta = ast.literal_eval(raw_meta)
                except Exception:
                    meta = {}
        elif isinstance(raw_meta, dict):
            meta = raw_meta
        else:
            meta = {}

        clean_meta = {
            key: (str(value) if value is not None else "")
            for key, value in meta.items()
        }
        documents.append(Document(page_content=chunk, metadata=clean_meta))

    if not documents:
        raise ValueError("No documents loaded from JSON - check your file path and schema.")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=chroma_dir,
    )
    logger.info("Chroma DB created with %d documents", len(documents))
    return vectorstore

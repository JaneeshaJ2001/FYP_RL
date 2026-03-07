# FYP_RL

Disaster-domain RAG chatbot focused on floods and landslides, built with LangGraph and Chroma.

## Project Structure

- `main.py`: application flow, nodes, graph wiring, chat loop
- `main.py`: CLI entry point and chat loop
- `nodes.py`: state definition, ingestion, node logic, tracing helpers
- `graph.py`: LangGraph construction and compilation
- `config.py`: centralized typed runtime config (`AppConfig`) + env overrides
- `prompts.py`: prompt templates used by generation/summarization nodes
- `chunks.json`: source chunk data for ingestion
- `chroma_db_disaster/`: persisted vector store

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and fill in keys.

3. Run the chatbot:

```bash
python main.py
```

4. Rebuild the vector store when needed:

```bash
python main.py --ingest --json chunks.json
```

## Tracing

Langfuse tracing is enabled automatically when `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` are set.

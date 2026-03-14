# Dataset Generation Pipeline

This dataset pipeline runs in 3 scripts and should be executed in this order.

## 1) Generate SOP Chunks

Run:

```bash
python chunk_dataset/chunks_generate.py
```

Result:

- `chunk_dataset/sop_chunks.json`

## 2) Group Topics and Add Labels

Run:

```bash
python chunk_dataset/topic_grouping.py
```

Result:

- `chunk_dataset/chunks_grouped.json`

## 3) Generate Multi-Turn Conversations

Run:

```bash
python dialog_dataset/generate_conversations.py
```

Result:

- `dialog_dataset/conversations.json`

## Quick Sequence

```bash
python chunk_dataset/chunks_generate.py
python chunk_dataset/topic_grouping.py
python dialog_dataset/generate_conversations.py
```
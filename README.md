# FYP_RL

Disaster-domain RAG chatbot focused on floods and landslides, with a reinforcement-learning-trained retrieval-decision layer.  Built with LangGraph, Stable-Baselines3, and Chroma.

## Project Structure

- `main.py`: CLI entry point and chat loop
- `rl_train.py`: PPO training + evaluation script
- `core/`: primary chatbot package
    - `config.py`: centralized typed runtime config (`AppConfig`) + env overrides
    - `prompts.py`: prompt templates used by generation/summarization nodes
    - `ingestion.py`: data loading and vectorstore creation/loading
    - `nodes.py`: state definition, node logic, retrieval/generation nodes
    - `graph.py`: LangGraph construction and compilation
    - `observability.py`: Langfuse tracing and run-config helpers
- `rl_core/`: RL-specific package
    - `state_encoder.py`: frozen transformer encoder (shared by policy + env)
    - `rl_policy.py`: custom SB3 `ActorCriticPolicy` (frozen backbone + MLP head)
    - `rl_env.py`: `RAGDecisionEnv` Gymnasium environment
    - `training_trace.py`: structured training trace logger
    - `utils.py`: `JudgeChain` + `approx_token_count`
- `dataset_generator/`: scripts and data for chunk/conversation dataset generation
- `training_logs/`: generated training trace logs

## Architecture

```
User query
    │
    ▼
[decide_retrieve]  ◄── RL policy (or baseline: always retrieve)
    │
    ├── action=1 ──► [retrieve]  ──► [generate_answer]
    └── action=0 ────────────────► [generate_answer]
                                         │
                                         ▼
                                [summarize_conversation]
                                         │
                                         ▼
                                       END
```

### RL Policy Architecture

```
Observation: encode_state(summary, query)
    = mean-pool( frozen-MiniLM("[summary] [SEP] [query]") )
    → float32 vector of shape (384,)

Policy network (only MLP is trained):
  [frozen encoder] → embedding(384)
        │
        ▼
  Linear(384→256) → ReLU → Linear(256→64) → ReLU
        │                                     │
        ▼ (actor head)                        ▼ (critic head)
  Linear(64→2) → action logits   Linear(64→1) → value
```

### Reward Signal

```
reward = judge_score(1–10)  −  β × approx_token_count(retrieved_docs)
```

The judge LLM rates the response against the ground truth on accuracy, safety, helpfulness, and empathy.  β=0.01 penalises unnecessary retrieval.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` from `.env.example` and fill in keys.

3. Run the baseline chatbot (always retrieve):
```bash
python main.py
```

4. Policy chatbot (RL-trained conditional retrieval):
```bash
python main.py --mode policy
```

5. Rebuild the vector store when needed:

```bash
python main.py --ingest --json dataset_generator/chunk_dataset/chunks_grouped.json
```

## Training the RL Policy

```bash
# Full training (20 000 steps default)
python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --timesteps 20000

# Quick smoke-test
python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --timesteps 500

# Evaluation only (requires saved policy)
python rl_train.py --episodes dataset_generator/dialog_dataset/conversations.json --eval-only \
    --policy-path policy_checkpoints/best_model.zip
```

TensorBoard logs land in `./tb_logs/`.  View with:
```bash
tensorboard --logdir tb_logs
```

## Tracing

Langfuse tracing is enabled automatically when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.
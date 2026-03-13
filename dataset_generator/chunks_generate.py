"""
SOP Chunk Dataset Generator
Generates 300 synthetic SOP chunks for RL retrieval-augmented disaster-response research.

Usage:
    pip install openai
    export OPENAI_API_KEY="your-api-key-here"
    python chunks_generate.py.py

Output:
    sop_chunks.json  — the full knowledge base ready for conversation generation
"""

import os
import json
import time
import math
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

TOTAL_CHUNKS    = 300
CHUNKS_PER_CALL = 30          # chunks requested per API call  (10 calls total)
OUTPUT_FILE     = "sop_chunks.json"
MODEL           = "gpt-4o-mini"    # or "gpt-4o-mini" for lower cost
MAX_TOKENS      = 8000

TOPICS = [
    "Flood preparedness",
    "Flood monitoring",
    "Flood safety",
    "Flood evacuation",
    "Flood Rescue Operations",
    "Post Flood Recovery",
    "Landslide risk assessment",
    "Landslide warning signs",
    "Landslide prevention",
    "Landslide Emergency Response",
    "Landslide Rescue Operations",
    "Post-Landslide Recovery",
    "Emergency communication",
    "general disaster preparedness",
    "emergency shelter",
    "medical emergency",
]

# Realistic official sources (varied across chunks)
SOURCES = [
    "FEMA Disaster Response Manual v2025",
    "WHO Emergency Field Operations Guide, 6th Edition",
    "National Disaster Management Authority SOP Compendium 2024",
    "IFRC Shelter & Settlement Guidelines 2024",
    "USGS Hazard Preparedness Handbook v4.1",
    "CDC Emergency Preparedness & Response SOP 2025",
    "UN OCHA Field Coordination Reference Card 2024",
    "Red Cross Disaster Response Operations Manual 2025",
    "NOAA Severe Weather Preparedness Guide 2024",
    "DHS National Response Framework 4th Edition",
    "EPA Chemical Emergency Preparedness Manual 2024",
    "ICRC First Aid & Field Medicine Handbook v3",
    "State Emergency Management Agency SOP Handbook 2025",
    "ISO 22320 Emergency Management Standard 2024 Implementation Guide",
]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(batch_topics: list[dict]) -> str:
    """
    Build the generation prompt for a single batch.

    batch_topics: list of {"topic": str, "chunk_ids": [str, ...], "source": str}
    """
    topic_lines = "\n".join(
        f"  - Topic: \"{t['topic']}\" | IDs: {t['chunk_ids']} | Source: \"{t['source']}\""
        for t in batch_topics
    )
    chunk_ids_flat = [cid for t in batch_topics for cid in t["chunk_ids"]]
    n = len(chunk_ids_flat)

    return f"""You are an expert synthetic dataset generator disaster-response dialogues.

## TASK
Generate exactly {n} high-quality synthetic SOP (Standard Operating Procedure) chunks for a \
disaster-response knowledge base. These chunks will later be used as the retrieval corpus for \
training a retrieval policy in a multi-turn conversational RL system.

## STRICT REQUIREMENTS
1. Return ONLY valid JSON — no markdown fences, no explanations, no extra text.
2. Each chunk must have exactly these four fields and no others:
   {{
     "chunk_id": "SOP_XXX",
     "text":     "...",
     "topic":    "...",
     "source":   "..."
   }}
3. Text length: 150–400 words per chunk. Be precise and detailed — this is real SOP content.
4. Every chunk must be unique, self-contained, and internally consistent.
5. Content must sound like authentic, professionally written emergency-management SOPs:
   - Reference realistic thresholds, timelines, equipment, and protocols
   - Include actionable directives, not vague advice
6. Use the exact chunk_ids, topics, and sources specified below.

## BATCH SPECIFICATION
{topic_lines}

## OUTPUT FORMAT
Return exactly this JSON structure:
{{
  "chunks": [
    {{ "chunk_id": "...", "text": "...", "topic": "...", "source": "..." }},
    ...
  ]
}}

Generate all {n} chunks now."""


# ── Batch planner ──────────────────────────────────────────────────────────────

def plan_batches(total: int, per_call: int) -> list[list[dict]]:
    """Divide chunk_ids across API calls, distributing topics evenly."""
    import random, itertools

    # Assign a topic to each chunk id sequentially (round-robin for even distribution)
    topic_cycle = itertools.cycle(TOPICS)
    assignments = []  # (chunk_id_str, topic, source)
    for i in range(1, total + 1):
        chunk_id = f"SOP_{i:03d}"
        topic    = next(topic_cycle)
        source   = SOURCES[(i - 1) % len(SOURCES)]
        assignments.append((chunk_id, topic, source))

    # Group into batches
    batches = []
    for batch_start in range(0, total, per_call):
        batch_assignments = assignments[batch_start : batch_start + per_call]
        # Group by (topic, source) within the batch for the prompt
        batch_topics = [
            {"topic": topic, "chunk_ids": [cid], "source": source}
            for cid, topic, source in batch_assignments
        ]
        batches.append(batch_topics)

    return batches


# ── API caller ─────────────────────────────────────────────────────────────────

def call_gpt(client: OpenAI, prompt: str, batch_num: int) -> list[dict]:
    """Call the OpenAI API and return parsed chunks."""
    print(f"  → Calling API for batch {batch_num}...")

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},   # enforces JSON output
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert synthetic dataset generator for disaster-response research. "
                    "Always respond with valid JSON only — no markdown, no explanations."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences if present (belt-and-suspenders)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if raw.endswith("```"):
        raw = raw[: raw.rfind("```")].strip()

    parsed = json.loads(raw)
    chunks = parsed.get("chunks", parsed)  # tolerate top-level list fallback
    return chunks


# ── Validator ──────────────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"chunk_id", "text", "topic", "source"}

def validate_chunks(chunks: list[dict], expected_ids: set[str]) -> list[dict]:
    valid = []
    for c in chunks:
        if not REQUIRED_FIELDS.issubset(c.keys()):
            print(f"    ⚠  Skipping chunk missing fields: {c.get('chunk_id', '?')}")
            continue
        word_count = len(c["text"].split())
        if word_count < 100:
            print(f"    ⚠  Chunk {c['chunk_id']} too short ({word_count} words) — keeping anyway")
        valid.append({k: c[k] for k in REQUIRED_FIELDS})  # drop any extra fields
    return valid


# ── Metadata builder ───────────────────────────────────────────────────────────

def build_metadata(chunks: list[dict]) -> dict:
    topic_dist = {}
    total_words = 0
    for c in chunks:
        topic_dist[c["topic"]] = topic_dist.get(c["topic"], 0) + 1
        total_words += len(c["text"].split())
    avg_words = round(total_words / len(chunks)) if chunks else 0
    return {
        "total_chunks": len(chunks),
        "topic_distribution": topic_dist,
        "average_chunk_length_words": avg_words,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable not set.")

    client  = OpenAI(api_key=api_key)
    batches = plan_batches(TOTAL_CHUNKS, CHUNKS_PER_CALL)

    print(f"SOP Chunk Dataset Generator")
    print(f"Model: {MODEL}")
    print(f"Target: {TOTAL_CHUNKS} chunks across {len(batches)} API calls\n")

    all_chunks = []
    seen_ids   = set()

    for batch_num, batch_topics in enumerate(batches, start=1):
        expected_ids = {cid for t in batch_topics for cid in t["chunk_ids"]}
        prompt = build_prompt(batch_topics)

        try:
            raw_chunks = call_gpt(client, prompt, batch_num)
        except json.JSONDecodeError as e:
            print(f"  ✗  JSON parse error in batch {batch_num}: {e}")
            print("     Skipping batch — re-run to fill gaps.")
            continue
        except Exception as e:
            print(f"  ✗  API error in batch {batch_num}: {e}")
            raise

        valid = validate_chunks(raw_chunks, expected_ids)

        # Deduplicate by chunk_id
        for c in valid:
            if c["chunk_id"] not in seen_ids:
                all_chunks.append(c)
                seen_ids.add(c["chunk_id"])
            else:
                print(f"    ⚠  Duplicate chunk_id {c['chunk_id']} — skipping")

        print(f"  ✓  Batch {batch_num} complete: {len(valid)} chunks added "
              f"(total so far: {len(all_chunks)})")

        # Polite rate-limit pause between calls
        if batch_num < len(batches):
            time.sleep(1)

    # Sort by chunk_id for clean output
    all_chunks.sort(key=lambda c: c["chunk_id"])

    output = {
        "sop_chunks": all_chunks,
        "metadata": build_metadata(all_chunks),
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    meta = output["metadata"]
    print(f"\n{'='*50}")
    print(f"Done!  {meta['total_chunks']} chunks saved to {OUTPUT_FILE}")
    print(f"Average chunk length: {meta['average_chunk_length_words']} words")
    print(f"Topic distribution:")
    for topic, count in sorted(meta["topic_distribution"].items()):
        print(f"  {topic:<35} {count:>3} chunks")


if __name__ == "__main__":
    main()
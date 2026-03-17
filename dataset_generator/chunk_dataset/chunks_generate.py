"""
SOP Chunk Dataset Generator

Usage:
    pip install openai
    export OPENAI_API_KEY="your-api-key-here"
    python dataset_generator/chunk_dataset/chunks_generate.py

Output:
    dataset_generator/chunk_dataset/sop_chunks.json  — the full knowledge base ready for conversation generation
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

TOTAL_CHUNKS    = 4
CHUNKS_PER_CALL = 4          # chunks requested per API call  (10 calls total)
BASE_DIR        = Path(__file__).resolve().parent
OUTPUT_FILE     = BASE_DIR / "sop_chunks.json"
MODEL           = "gpt-4o-mini"    # or "gpt-4o"
MAX_TOKENS      = 8000

TOPICS = [
    "flood preparedness",
    "flood safety",
    "flood evacuation",
    "flood rescue operations",
    "post flood recovery",
    "flood monitoring",
    "landslide risk assessment",
    "landslide warning signs",
    "landslide prevention",
    "landslide emergency response",
    "landslide rescue operations",
    "post landslide recovery",
    "emergency communication",
    "general disaster preparedness",
    "emergency shelter",
    "medical emergency",
    "infrastructure damage",
    "search & rescue logistics",
    "community response behavior"
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

INFO_TYPES = [
    "general_knowledge",
    "hazard_explanation",
    "SOP_procedure",
    "rescue_guideline",
    "evacuation_plan",
    "emergency_alert",
    "medical_protocol",
    "safety_constraint",
    "vulnerable_group_guideline",
    "recovery_plan",
    "FAQ",
]

TOPIC_INFO_TYPE_CYCLE = {
    "flood preparedness": ["general_knowledge", "SOP_procedure", "FAQ"],
    "flood safety": ["safety_constraint", "general_knowledge", "FAQ"],
    "flood evacuation": ["evacuation_plan", "emergency_alert", "SOP_procedure"],
    "flood rescue operations": ["rescue_guideline", "SOP_procedure", "medical_protocol"],
    "post flood recovery": ["recovery_plan", "FAQ", "general_knowledge"],
    "flood monitoring": ["hazard_explanation", "emergency_alert"],    "landslide risk assessment": ["hazard_explanation", "SOP_procedure"],
    "landslide warning signs": ["hazard_explanation", "emergency_alert"],
    "landslide prevention": ["general_knowledge", "safety_constraint", "SOP_procedure"],
    "landslide emergency response": ["SOP_procedure", "emergency_alert", "rescue_guideline"],
    "landslide rescue operations": ["rescue_guideline", "medical_protocol", "SOP_procedure"],
    "post landslide recovery": ["recovery_plan", "FAQ", "general_knowledge"],
    "emergency communication": ["emergency_alert", "SOP_procedure"],
    "general disaster preparedness": ["general_knowledge", "SOP_procedure", "FAQ", "vulnerable_group_guideline"],
    "emergency shelter": ["SOP_procedure", "vulnerable_group_guideline", "safety_constraint"],
    "medical emergency": ["medical_protocol", "SOP_procedure", "FAQ"],
    "infrastructure damage": ["safety_constraint", "recovery_plan"],
    "search & rescue logistics": ["rescue_guideline", "SOP_procedure"],
    "community response behavior": ["general_knowledge", "FAQ", "vulnerable_group_guideline", "safety_constraint"],
}


def assign_information_type(topic: str, topic_counter: dict[str, int]) -> str:
    """Assign deterministic information_type for each chunk based on its topic."""
    cycle = TOPIC_INFO_TYPE_CYCLE.get(topic, INFO_TYPES)
    index = topic_counter.get(topic, 0)
    topic_counter[topic] = index + 1
    return cycle[index % len(cycle)]

# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(batch_chunks: list[dict]) -> str:
    """
    Build the generation prompt for a single batch.

    batch_chunks: list of {
        "chunk_id": str,
        "topic": str,
        "source": str,
        "information_type": str,
    }
    """
    chunk_lines = "\n".join(
        f"  - Chunk ID: \"{c['chunk_id']}\" | Topic: \"{c['topic']}\" | Source: \"{c['source']}\" | Information Type: \"{c['information_type']}\""
        for c in batch_chunks
    )
    info_type_lines = "\n".join(f"- {info_type}" for info_type in INFO_TYPES)
    n = len(batch_chunks)

    return f"""You are an expert synthetic dataset generator disaster-response knowledge systems.

## TASK
Generate exactly {n} high-quality knowledge chunks for a disaster-response AI assistant. \
These chunks will form a retrieval corpus for a multi-turn conversational system. \
The assistant interacts primarily with civilians during floods and landslides, but also reflects professional disaster-management protocols.

## STRICT OUTPUT FORMAT
1. Return ONLY valid JSON — no markdown fences, no explanations, no extra text.
2. Each chunk must have exactly these five fields and no others:
   {{
     "chunk_id": "SOP_XXX",
     "text":     "...",
     "topic":    "...",
     "source":   "...",
     "information_type": "..."
   }}

## INFORMATION TYPE REQUIREMENT
Each chunk MUST belong to exactly ONE of the following categories:

{info_type_lines}

## WRITING STYLE RULES (VERY IMPORTANT)

The writing style MUST strictly match the information_type:

1. general_knowledge  
   - Explanatory, descriptive  
   - Define concepts clearly  
   - Example: “A landslide occurs when…”

2. hazard_explanation  
   - Cause-effect reasoning  
   - Explain triggers and mechanisms  
   - Example: rainfall → soil saturation → slope failure  

3. SOP_procedure  
   - Step-by-step structured instructions  
   - Include timing, thresholds, coordination  

4. rescue_guideline  
   - Life-saving actions  
   - Search, rescue, survival techniques  

5. evacuation_plan  
   - Clear movement instructions  
   - Routes, triggers, safe locations  

6. emergency_alert  
   - Urgent, short, directive  
   - Warning tone  
   - Example: “Evacuate immediately…” 

7. medical_protocol  
   - First aid and medical procedures  
   - Clear step sequences  

8. safety_constraint  
   - Focus on what NOT to do  
   - Prevent common mistakes  

9. vulnerable_group_guideline  
   - Special instructions for children, elderly, disabled  

10. recovery_plan  
   - Post-disaster rebuilding and restoration  

11. FAQ  
   - Short, direct answers to common civilian questions

## CONTENT REQUIREMENTS

1. Length: 200–400 words per chunk  
2. Each chunk must be:
   - Self-contained  
   - Internally consistent  
   - Focused on a single idea  

3. Must reflect realistic disaster-response knowledge:
   - Include measurable thresholds (e.g., rainfall levels, water depth)
   - Mention real-world actions (evacuation timing, equipment, coordination)
   - Include practical, actionable guidance where applicable  

4. Civilians must be able to understand and use the content  
5. Avoid purely administrative/internal procedures  

## DIVERSITY REQUIREMENTS

Ensure strong variation across:

- Different information_types
- Different tones (urgent, explanatory, procedural)
- Different perspectives (civilian vs responder guidance)

Do NOT repeat similar content across chunks.

## RETRIEVAL-AWARE DESIGN

The dataset must support learning when to retrieve information.
Therefore:
- Some chunks should be highly actionable (e.g., evacuation steps)
- Some should be descriptive (e.g., definitions)
- Some should be situational (e.g., current conditions)
- Some should be constraint-based (what NOT to do)

Avoid making all chunks equally procedural.

Use the exact chunk_ids, topics, sources, and information_types specified below.

## BATCH SPECIFICATION
{chunk_lines}

Each line specifies:
- chunk_id
- topic
- source
- information_type

You MUST use the exact values provided.

Additionally:
- Use the exact information_type assigned to each chunk_id (do not change it)
- Ensure the text style matches the assigned information_type

## OUTPUT FORMAT
Return exactly this JSON structure:
{{
  "chunks": [
    {{ "chunk_id": "...", "text": "...", "topic": "...", "source": "...", "information_type": "..." }},
    ...
  ]
}}

Generate all {n} chunks now."""


# ── Batch planner ──────────────────────────────────────────────────────────────

def plan_batches(total: int, per_call: int) -> list[list[dict]]:
    """Divide chunk specs across API calls with deterministic information_type labels."""
    import itertools

    # Assign a topic to each chunk id sequentially (round-robin for even distribution)
    topic_cycle = itertools.cycle(TOPICS)
    topic_counter: dict[str, int] = {}
    assignments = []
    for i in range(1, total + 1):
        chunk_id = f"SOP_{i:03d}"
        topic    = next(topic_cycle)
        source   = SOURCES[(i - 1) % len(SOURCES)]
        information_type = assign_information_type(topic, topic_counter)
        assignments.append(
            {
                "chunk_id": chunk_id,
                "topic": topic,
                "source": source,
                "information_type": information_type,
            }
        )

    # Group into batches
    batches = []
    for batch_start in range(0, total, per_call):
        batch_assignments = assignments[batch_start : batch_start + per_call]
        batches.append(batch_assignments)

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

REQUIRED_INPUT_FIELDS = {"chunk_id", "text"}

def validate_chunks(chunks: list[dict], expected_specs: dict[str, dict]) -> list[dict]:
    valid = []
    missing_ids = set(expected_specs)
    for c in chunks:
        if not REQUIRED_INPUT_FIELDS.issubset(c.keys()):
            print(f"    ⚠  Skipping chunk missing fields: {c.get('chunk_id', '?')}")
            continue

        chunk_id = str(c["chunk_id"])
        expected = expected_specs.get(chunk_id)
        if expected is None:
            print(f"    ⚠  Unexpected chunk_id {chunk_id} — skipping")
            continue

        text = str(c["text"]).strip()
        word_count = len(text.split())
        if word_count < 100:
            print(f"    ⚠  Chunk {chunk_id} too short ({word_count} words) — keeping anyway")

        valid.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "topic": expected["topic"],
                "source": expected["source"],
                "information_type": expected["information_type"],
            }
        )
        missing_ids.discard(chunk_id)

    for chunk_id in sorted(missing_ids):
        print(f"    ⚠  Missing chunk in response: {chunk_id}")

    return valid


# ── Metadata builder ───────────────────────────────────────────────────────────

def build_metadata(chunks: list[dict]) -> dict:
    topic_dist = {}
    info_type_dist = {}
    total_words = 0
    for c in chunks:
        topic_dist[c["topic"]] = topic_dist.get(c["topic"], 0) + 1
        info_type = c.get("information_type", "unknown")
        info_type_dist[info_type] = info_type_dist.get(info_type, 0) + 1
        total_words += len(c["text"].split())
    avg_words = round(total_words / len(chunks)) if chunks else 0
    return {
        "total_chunks": len(chunks),
        "topic_distribution": topic_dist,
        "information_type_distribution": info_type_dist,
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

    for batch_num, batch_chunks in enumerate(batches, start=1):
        expected_specs = {c["chunk_id"]: c for c in batch_chunks}
        prompt = build_prompt(batch_chunks)

        try:
            raw_chunks = call_gpt(client, prompt, batch_num)
        except json.JSONDecodeError as e:
            print(f"  ✗  JSON parse error in batch {batch_num}: {e}")
            print("     Skipping batch — re-run to fill gaps.")
            continue
        except Exception as e:
            print(f"  ✗  API error in batch {batch_num}: {e}")
            raise

        valid = validate_chunks(raw_chunks, expected_specs)

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
    print(f"Information type distribution:")
    for info_type, count in sorted(meta["information_type_distribution"].items()):
        print(f"  {info_type:<35} {count:>3} chunks")


if __name__ == "__main__":
    main()
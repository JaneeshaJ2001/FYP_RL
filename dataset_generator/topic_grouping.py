import json
import re
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = "sop_chunks.json"
OUTPUT_PATH = "chunks_grouped.json"
UNMAPPED_PATH = "unmapped_topics.json"


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


TOPIC_MAP = {
    "flood preparedness": {
        "hazard_type": "flood",
        "topic_group": "preparedness",
        "stage": "before_disaster",
    },
    "flood monitoring": {
        "hazard_type": "flood",
        "topic_group": "monitoring_warning",
        "stage": "before_disaster",
    },
    "flood evacuation": {
        "hazard_type": "flood",
        "topic_group": "evacuation",
        "stage": "during_disaster",
    },
    "flood safety": {
        "hazard_type": "flood",
        "topic_group": "safety_response",
        "stage": "during_disaster",
    },
    "flood rescue operations": {
        "hazard_type": "flood",
        "topic_group": "rescue_operations",
        "stage": "during_disaster",
    },
    "post flood recovery": {
        "hazard_type": "flood",
        "topic_group": "recovery",
        "stage": "after_disaster",
    },
    "landslide risk assessment": {
        "hazard_type": "landslide",
        "topic_group": "monitoring_warning",
        "stage": "before_disaster",
    },
    "landslide warning signs": {
        "hazard_type": "landslide",
        "topic_group": "monitoring_warning",
        "stage": "before_disaster",
    },
    "landslide prevention": {
        "hazard_type": "landslide",
        "topic_group": "preparedness",
        "stage": "before_disaster",
    },
    "landslide emergency response": {
        "hazard_type": "landslide",
        "topic_group": "safety_response",
        "stage": "during_disaster",
    },
    "landslide rescue operations": {
        "hazard_type": "landslide",
        "topic_group": "rescue_operations",
        "stage": "during_disaster",
    },
    "post landslide recovery": {
        "hazard_type": "landslide",
        "topic_group": "recovery",
        "stage": "after_disaster",
    },
    "emergency communication": {
        "hazard_type": "multi_hazard",
        "topic_group": "communication_coordination",
        "stage": "cross_cutting",
    },
    "general disaster preparedness": {
        "hazard_type": "multi_hazard",
        "topic_group": "preparedness",
        "stage": "before_disaster",
    },
    "emergency shelter": {
        "hazard_type": "multi_hazard",
        "topic_group": "shelter_management",
        "stage": "during_disaster",
    },
    "medical emergency": {
        "hazard_type": "multi_hazard",
        "topic_group": "medical_response",
        "stage": "during_disaster",
    },
}


ALIASES = {
    "post landslide recovery": "post landslide recovery",
    "post-landslide recovery": "post landslide recovery",
    "post flood recovery": "post flood recovery",
    "flood rescue operation": "flood rescue operations",
    "landslide rescue operation": "landslide rescue operations",
    "landslide emergency response": "landslide emergency response",
    "emergency communication": "emergency communication",
    "general disaster preparedness": "general disaster preparedness",
    "medical emergency": "medical emergency",
    "emergency shelter": "emergency shelter",
}


def canonicalize_topic(raw_topic: str) -> str:
    normalized = normalize_text(raw_topic)
    return ALIASES.get(normalized, normalized)


def extract_topic(chunk: dict) -> str:
    # Adjust this if your key is different
    return (
        chunk.get("topic")
        or chunk.get("metadata", {}).get("topic")
        or chunk.get("category")
        or ""
    )


def enrich_chunk(chunk: dict) -> tuple[dict, str | None]:
    raw_topic = extract_topic(chunk)
    normalized_topic = canonicalize_topic(raw_topic)

    chunk["raw_topic"] = raw_topic
    chunk["topic_normalized"] = normalized_topic

    mapping = TOPIC_MAP.get(normalized_topic)
    if mapping is None:
        chunk["hazard_type"] = "unknown"
        chunk["topic_group"] = "unknown"
        chunk["stage"] = "unknown"
        return chunk, normalized_topic

    chunk["hazard_type"] = mapping["hazard_type"]
    chunk["topic_group"] = mapping["topic_group"]
    chunk["stage"] = mapping["stage"]
    return chunk, None


def print_counter(title: str, counter: Counter):
    print(f"\n{title}")
    if not counter:
        print("  (none)")
        return
    max_key_len = max(len(str(k)) for k in counter)
    for key, value in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {str(key):<{max_key_len}}  {value}")


def main():
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Support wrapped payloads like {"sop_chunks": [...]}.
        for key in ("sop_chunks", "chunks", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                data = candidate
                break

    if not isinstance(data, list):
        raise ValueError(
            "Expected sop_chunks.json to contain a JSON list or an object with a list under one of: sop_chunks, chunks, data"
        )

    enriched = []
    unmapped_counter = Counter()

    raw_topic_counter = Counter()
    normalized_topic_counter = Counter()
    hazard_counter = Counter()
    group_counter = Counter()
    stage_counter = Counter()
    combined_counter = Counter()

    for chunk in data:
        raw_topic = extract_topic(chunk)
        raw_topic_counter[raw_topic] += 1

        updated_chunk, unmapped = enrich_chunk(chunk)
        enriched.append(updated_chunk)

        normalized_topic_counter[updated_chunk["topic_normalized"]] += 1
        hazard_counter[updated_chunk["hazard_type"]] += 1
        group_counter[updated_chunk["topic_group"]] += 1
        stage_counter[updated_chunk["stage"]] += 1
        combined_counter[
            f'{updated_chunk["hazard_type"]} / {updated_chunk["topic_group"]}'
        ] += 1

        if unmapped:
            unmapped_counter[unmapped] += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    with open(UNMAPPED_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"topic_normalized": k, "count": v} for k, v in sorted(unmapped_counter.items())],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nProcessed {len(enriched)} chunks")
    print(f"Saved grouped chunks to: {OUTPUT_PATH}")
    print(f"Saved unmapped topics to: {UNMAPPED_PATH}")

    print_counter("Raw topic distribution", raw_topic_counter)
    print_counter("Normalized topic distribution", normalized_topic_counter)
    print_counter("Hazard type distribution", hazard_counter)
    print_counter("Topic group distribution", group_counter)
    print_counter("Stage distribution", stage_counter)
    print_counter("Hazard / Topic group distribution", combined_counter)

    if unmapped_counter:
        print_counter("Unmapped normalized topics", unmapped_counter)
        print("\nAdd these unmapped topics into ALIASES or TOPIC_MAP.")
    else:
        print("\nAll topics mapped successfully.")


if __name__ == "__main__":
    main()
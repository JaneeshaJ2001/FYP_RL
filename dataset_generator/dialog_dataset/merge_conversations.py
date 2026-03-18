"""
merge_conversations.py
──────────────────────
Merges multiple conversations JSON files (same structure) into one.

  • Episodes are renumbered sequentially (conv_001, conv_002, …) in the
    order the input files are given — no conversations are reordered or dropped.
  • Metadata numeric fields are summed; derived fields (avg, rates) are
    recomputed from the merged totals.
  • Metadata dict fields (scenario_distribution, topic_distribution,
    information_type_coverage, query_type_distribution) are merged by
    summing counts; new keys that appear in any file are included.
  • Any unexpected top-level metadata field is carried forward (summed if
    numeric, last-value-wins if not).

Usage:
    python merge_conversations.py file1.json file2.json file3.json
    python merge_conversations.py file1.json file2.json --output merged.json
    python merge_conversations.py *.json --output dataset/merged.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ── Fields that are plain sums ─────────────────────────────────────────────────
ADDITIVE_SCALAR_FIELDS = {
    "total_episodes",
    "total_turns",
    "retrieval_true_turns",
    "retrieval_false_turns",
}

# ── Fields that are dicts of counts (merged by summing per-key) ────────────────
ADDITIVE_DICT_FIELDS = {
    "scenario_distribution",
    "topic_distribution",
    "information_type_coverage",
    "query_type_distribution",
}

# ── Fields recomputed from merged totals ───────────────────────────────────────
DERIVED_FIELDS = {
    "avg_turns_per_episode",
    "retrieval_required_rate",
}


def merge_metadata(metas: list[dict]) -> dict:
    merged: dict = {}
    extra_scalars: dict[str, float] = defaultdict(float)
    extra_dicts:   dict[str, dict]  = defaultdict(lambda: defaultdict(float))

    # Accumulate
    for meta in metas:
        for key, value in meta.items():
            if key in ADDITIVE_SCALAR_FIELDS:
                merged[key] = merged.get(key, 0) + value

            elif key in ADDITIVE_DICT_FIELDS:
                bucket = merged.setdefault(key, defaultdict(int))
                if isinstance(value, dict):
                    for sub_key, count in value.items():
                        bucket[sub_key] += count

            elif key in DERIVED_FIELDS:
                pass  # will be recomputed below

            else:
                # Unknown field: sum if numeric, last-value-wins otherwise
                if isinstance(value, (int, float)):
                    extra_scalars[key] += value
                else:
                    merged[key] = value

    # Convert defaultdicts back to plain dicts
    for field in ADDITIVE_DICT_FIELDS:
        if field in merged:
            merged[field] = dict(merged[field])

    # Merge unknown numeric extras
    for key, value in extra_scalars.items():
        merged[key] = value

    # Recompute derived fields
    total_eps   = merged.get("total_episodes", 0)
    total_turns = merged.get("total_turns", 0)
    ret_true    = merged.get("retrieval_true_turns", 0)

    merged["avg_turns_per_episode"]   = round(total_turns / total_eps, 2) if total_eps else 0
    merged["retrieval_required_rate"] = round(ret_true / total_turns, 3)  if total_turns else 0

    return merged


def merge_files(input_paths: list[Path], output_path: Path) -> None:
    all_conversations: list[dict] = []
    all_metadata:      list[dict] = []

    for path in input_paths:
        print(f"  Reading {path} ...", end=" ")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        conversations = data.get("conversations")
        if not isinstance(conversations, list):
            raise ValueError(f"{path}: expected a top-level 'conversations' list.")

        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            all_metadata.append(metadata)
        else:
            print(f"(no metadata found, skipping metadata from this file)")

        all_conversations.extend(conversations)
        print(f"{len(conversations)} episodes")

    # Renumber sequentially in the order they were added
    for idx, episode in enumerate(all_conversations, start=1):
        episode["episode_id"] = f"conv_{idx:03d}"

    merged_metadata = merge_metadata(all_metadata) if all_metadata else {}

    output = {
        "conversations": all_conversations,
        "metadata":      merged_metadata,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'─' * 56}")
    print(f"  Merged {len(all_conversations)} episodes → {output_path}")
    print(f"  Total turns      : {merged_metadata.get('total_turns', '?')}")
    print(f"  Avg turns/ep     : {merged_metadata.get('avg_turns_per_episode', '?')}")
    print(f"  Retrieval rate   : {merged_metadata.get('retrieval_required_rate', '?')}")
    print(f"\n  Scenario distribution:")
    for sc, cnt in sorted(merged_metadata.get("scenario_distribution", {}).items()):
        print(f"    {sc}: {cnt}")
    print(f"{'─' * 56}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge multiple conversations JSON files")
    p.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON files (processed in the order given)",
    )
    p.add_argument(
        "--output",
        default="merged_conversations.json",
        help="Output file path (default: merged_conversations.json)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]

    missing = [p for p in input_paths if not p.exists()]
    if missing:
        raise SystemExit(f"ERROR: file(s) not found: {', '.join(str(p) for p in missing)}")

    merge_files(input_paths, Path(args.output))
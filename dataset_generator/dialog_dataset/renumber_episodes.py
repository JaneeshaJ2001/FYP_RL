"""
renumber_episodes.py
────────────────────
Reads conversations.json, renumbers each episode_id sequentially
(conv_001, conv_002, ...) in the order they appear, and writes the
result back out without touching any other field.

Usage:
    python renumber_episodes.py                          # in-place
    python renumber_episodes.py --input conversations_A.json --output conversations_A_new.json
"""

import argparse
import json
from pathlib import Path


def renumber(input_path: Path, output_path: Path) -> None:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Expected a top-level 'conversations' list.")

    for idx, episode in enumerate(conversations, start=1):
        episode["episode_id"] = f"conv_{idx:03d}"

    # Rebuild metadata total_episodes count if the key exists
    if "metadata" in data and isinstance(data["metadata"], dict):
        data["metadata"]["total_episodes"] = len(conversations)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Renumbered {len(conversations)} episodes → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Renumber episode_ids sequentially")
    p.add_argument("--input",  default="conversations.json")
    p.add_argument("--output", default=None,
                   help="Defaults to overwriting the input file")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path
    renumber(input_path, output_path)
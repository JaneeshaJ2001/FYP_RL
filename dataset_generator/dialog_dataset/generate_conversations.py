"""
Multi-Turn Disaster Dialogue Generator
====================================================================
Generates conversation episodes grounded in chunks_grouped.json.
Supports 14 scenario templates (A–N).

Single-pass pipeline
────────────────────
One API call per episode: the model generates the conversation AND annotates
every turn (retrieval_required, relevant_chunks, query_type) together.

Scope
─────
• Floods and landslides ONLY.
• Single-hazard episodes; Scenario H is the sole cross-hazard comparison.
• OOD = unsupported disaster type OR unsupported local scope.
  NO unrelated non-disaster topics.

Usage
─────
    pip install openai
    export OPENAI_API_KEY="sk-..."

    # First batch – 20 episodes
    python generate_conversations.py --episodes 20

    # Custom output path
    python generate_conversations.py --episodes 100 --output my_data/convs.json
"""

import argparse
import os
import json
import time
import random
from collections import defaultdict
from pathlib import Path
from openai import OpenAI

from conversation_prompts import (
    DATASET_SYSTEM_PROMPT,
    VALID_QUERY_TYPES,
    RETRIEVAL_FALSE_TYPES,
    RETRIEVAL_TRUE_TYPES,
    build_prompt,
    get_scenario_template,
)

# ── Default configuration ──────────────────────────────────────────────────────

DEFAULT_INPUT_FILE  = Path(__file__).resolve().parent.parent / "chunk_dataset" / "chunks_grouped.json"
DEFAULT_OUTPUT_FILE = Path(__file__).resolve().parent / "conversations.json"
MODEL               = "gpt-4o-mini"
MAX_TOKENS          = 3500
CHUNKS_PER_EP       = 6     # primary-hazard chunks per single-hazard episode
MH_SUPPLEMENT       = 2     # multi_hazard chunks blended into every single-hazard episode
SLEEP_BETWEEN       = 0.5   # seconds between API calls

# ── Scenario distribution ──────────────────────────────────────────────────────
# 
#   A  Follow-up Conversation              6 turns
#   B  Information Seeking                 5 turns
#   C  Emergency Dialogue                  6 turns
#   D  Subtopic Shift Within Hazard        6 turns
#   E  Ambiguous / Non-Standalone Query    6 turns
#   F  Unanswerable / Partial Answer       6 turns
#   G  Lifecycle Progression               6 turns
#   H  Comparative: Flood vs Landslide     6 turns  (cross-hazard)
#   I  Unsupported Scope / Out-of-Domain   6 turns
#   J  Misconception / Correction          5 turns
#   K  Clarification Needed Before Answer  5 turns
#   L  Multi-Chunk Synthesis               6 turns
#   M  Summary / Checklist From Context    5 turns
#   N  Apparent Conflict / Resolution      6 turns

SCENARIO_DIST = {
    # "A": 350,
    # "B": 350,
    # "C": 350,
    # "D": 350,
    # "E": 350,
    # "F": 350,
    # "G": 350,
    # "H": 350,
    # "I": 350,
    # "J": 350,
    # "K": 350,
    # "L": 350,
    # "M": 350,
    # "N": 350,
}

DEFAULT_TOTAL = sum(SCENARIO_DIST.values())


# ── Load & index chunks ────────────────────────────────────────────────────────

def load_chunks(path: str | Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = None
        for key in ("sop_chunks", "chunks"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                chunks = candidate
                break
        if chunks is None:
            raise ValueError(
                "Expected chunk input to be a list or an object "
                "containing a list under 'sop_chunks' or 'chunks'."
            )
    else:
        raise ValueError("Expected chunk input to be a list or object.")

    index        = defaultdict(list)   # (hazard, group) → chunks
    hazard_index = defaultdict(list)   # hazard → chunks

    for c in chunks:
        hazard = c.get("hazard_type") or c.get("topic", "general").split()[0]
        group  = c.get("topic_group") or c.get("topic", "general")
        index[(hazard, group)].append(c)
        hazard_index[hazard].append(c)

    return index, hazard_index, chunks


# ── Chunk sampler ──────────────────────────────────────────────────────────────

def sample_diverse_by_information_type(pool: list[dict], sample_size: int) -> list[dict]:
    """Sample chunks while encouraging information_type diversity."""
    if sample_size <= 0:
        return []
    if len(pool) <= sample_size:
        return list(pool)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for chunk in pool:
        by_type[chunk.get("information_type", "unknown")].append(chunk)

    for chunks_for_type in by_type.values():
        random.shuffle(chunks_for_type)

    selected: list[dict] = []
    info_types = list(by_type.keys())
    random.shuffle(info_types)

    # First pass: one chunk per information_type
    for info_type in info_types:
        bucket = by_type[info_type]
        if bucket:
            selected.append(bucket.pop())
        if len(selected) == sample_size:
            return selected

    # Second pass: fill remaining slots from leftovers
    leftovers: list[dict] = []
    for bucket in by_type.values():
        leftovers.extend(bucket)
    random.shuffle(leftovers)
    selected.extend(leftovers[: sample_size - len(selected)])
    return selected


def collect_information_types(chunks: list[dict]) -> list[str]:
    return sorted({str(c.get("information_type", "unknown")) for c in chunks})


def _sample_multi_hazard_supplement(
    hazard_index: dict,
    primary_group: str,
    n: int,
) -> list[dict]:
    """
    Return up to `n` multi_hazard chunks, preferring those whose topic_group
    matches the primary episode group for topical relevance.
    """
    if n <= 0:
        return []
    pool = hazard_index.get("multi_hazard", [])
    if not pool:
        return []
    matched   = [c for c in pool if c.get("topic_group", "") == primary_group]
    unmatched = [c for c in pool if c.get("topic_group", "") != primary_group]
    return sample_diverse_by_information_type(
        matched + unmatched, min(n, len(pool))
    )


def sample_chunks_for_episode(
    scenario: str,
    index: dict,
    hazard_index: dict,
    single_hazard_keys: list,
) -> tuple[list[dict], str, str]:
    """
    Build the chunk set for one episode.

    Scenario H (Comparative):
        2–3 flood chunks  +  2–3 landslide chunks  +  1–2 multi_hazard chunks.

    All other scenarios (single-hazard):
        Up to (CHUNKS_PER_EP - MH_SUPPLEMENT) primary-hazard chunks
        + up to MH_SUPPLEMENT multi_hazard chunks blended in.

    multi_hazard chunks are labelled hazard_type=multi_hazard in the chunk
    block so the LLM knows they apply to both hazards.
    """
    if scenario == "H":
        flood_chunks     = hazard_index.get("flood", [])
        landslide_chunks = hazard_index.get("landslide", [])
        if not flood_chunks or not landslide_chunks:
            available = [h for h in hazard_index if h != "multi_hazard" and hazard_index[h]]
            if len(available) >= 2:
                h1, h2 = random.sample(available, 2)
            else:
                h1 = h2 = available[0]
            flood_chunks     = hazard_index[h1]
            landslide_chunks = hazard_index[h2]
            hazard_label = f"{h1}+{h2}"
        else:
            hazard_label = "flood+landslide"

        selected = (
            sample_diverse_by_information_type(flood_chunks,     min(3, len(flood_chunks)))
            + sample_diverse_by_information_type(landslide_chunks, min(3, len(landslide_chunks)))
            + _sample_multi_hazard_supplement(hazard_index, "cross_hazard_comparison", 2)
        )
        return selected, hazard_label, "cross_hazard_comparison"

    # Single-hazard episode
    key    = random.choice(single_hazard_keys)
    pool   = index[key]
    hazard, group = key
    n_primary      = max(1, CHUNKS_PER_EP - MH_SUPPLEMENT)
    primary_chunks = sample_diverse_by_information_type(pool, min(n_primary, len(pool)))
    mh_chunks      = _sample_multi_hazard_supplement(hazard_index, group, MH_SUPPLEMENT)
    return primary_chunks + mh_chunks, hazard, group


# ── Episode planner ────────────────────────────────────────────────────────────

def plan_episodes(index: dict, hazard_index: dict, scenario_dist: dict) -> list[dict]:
    # Only use flood/landslide keys as episode anchors; multi_hazard is supplementary
    single_hazard_keys = [k for k in index.keys() if k[0] != "multi_hazard"]

    scenarios: list[str] = []
    for sc, cnt in scenario_dist.items():
        scenarios.extend([sc] * cnt)
    random.shuffle(scenarios)

    episodes: list[dict] = []
    for i, sc in enumerate(scenarios):
        selected, hazard, group = sample_chunks_for_episode(
            sc, index, hazard_index, single_hazard_keys
        )
        episodes.append({
            "episode_id"       : f"conv_{i + 1:03d}",
            "scenario"         : sc,
            "primary_topic"    : f"{hazard} / {group}",
            "hazard_type"      : hazard,
            "topic_group"      : group,
            "information_types": collect_information_types(selected),
            "selected_chunks"  : selected,
        })
    return episodes


# ── API call ───────────────────────────────────────────────────────────────────

def call_gpt(client: OpenAI, prompt: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DATASET_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    return json.loads(r.choices[0].message.content.strip())


# ── Validator ──────────────────────────────────────────────────────────────────

REQUIRED_TURN_FIELDS = {
    "turn_id", "query", "assistant_answer",
    "relevant_chunks", "retrieval_required", "query_type",
}


def validate_episode(ep: dict, expected_turns: int) -> tuple[bool, str]:
    turns = ep.get("turns", [])
    if len(turns) != expected_turns:
        return False, f"expected {expected_turns} turns, got {len(turns)}"

    seen_chunk_ids: set[str] = set()
    has_ret_true = has_ret_false = False

    for t in turns:
        tid = t.get("turn_id", "?")
        missing = REQUIRED_TURN_FIELDS - t.keys()
        if missing:
            return False, f"turn {tid} missing fields"

        qt = t["query_type"]
        if qt not in VALID_QUERY_TYPES:
            return False, f"turn {tid} unknown query_type '{qt}'"
        if not isinstance(t["retrieval_required"], bool):
            return False, f"turn {tid} retrieval_required is not bool"
        if not isinstance(t["relevant_chunks"], list):
            return False, f"turn {tid} relevant_chunks is not list"

        rr     = t["retrieval_required"]
        chunks = t["relevant_chunks"]

        if qt in RETRIEVAL_FALSE_TYPES and rr:
            return False, f"turn {tid} query_type='{qt}' requires retrieval_required=false"
        if qt in RETRIEVAL_TRUE_TYPES and not rr:
            return False, f"turn {tid} query_type='{qt}' requires retrieval_required=true"

        if rr:
            if not chunks:
                return False, f"turn {tid} retrieval_required=true but relevant_chunks=[]"
            new = set(chunks) - seen_chunk_ids
            if not new:
                return False, (
                    f"turn {tid} retrieval_required=true but all cited chunks "
                    "were already retrieved in earlier turns"
                )
            seen_chunk_ids.update(chunks)
            has_ret_true = True
        else:
            if chunks:
                return False, f"turn {tid} retrieval_required=false but relevant_chunks is non-empty"
            has_ret_false = True

    if not has_ret_true:
        return False, "episode has no turn with retrieval_required=true"
    if not has_ret_false:
        return False, "episode has no turn with retrieval_required=false"
    return True, "ok"


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_metadata(episodes: list[dict]) -> dict:
    sc_dist            = defaultdict(int)
    qt_dist            = defaultdict(int)
    top_dist           = defaultdict(int)
    info_type_coverage = defaultdict(int)
    ret_true = ret_false = total_t = 0

    for ep in episodes:
        sc_dist[ep.get("scenario_type", "?")] += 1
        top_dist[ep.get("primary_topic", "?")] += 1
        for it in ep.get("information_types", []):
            info_type_coverage[it] += 1
        for t in ep.get("turns", []):
            total_t += 1
            qt_dist[t["query_type"]] += 1
            if t.get("retrieval_required"):
                ret_true += 1
            else:
                ret_false += 1

    return {
        "total_episodes"           : len(episodes),
        "total_turns"              : total_t,
        "avg_turns_per_episode"    : round(total_t / len(episodes), 2) if episodes else 0,
        "retrieval_required_rate"  : round(ret_true / total_t, 3) if total_t else 0,
        "retrieval_true_turns"     : ret_true,
        "retrieval_false_turns"    : ret_false,
        "scenario_distribution"    : dict(sc_dist),
        "topic_distribution"       : dict(top_dist),
        "information_type_coverage": dict(info_type_coverage),
        "query_type_distribution"  : dict(qt_dist),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-turn disaster dialogue dataset"
    )
    p.add_argument(
        "--input", default=str(DEFAULT_INPUT_FILE),
        help="Path to chunks_grouped.json",
    )
    p.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_FILE),
        help="Output conversations.json path",
    )
    p.add_argument(
        "--episodes", type=int, default=DEFAULT_TOTAL,
        help="Total episodes to generate",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return p.parse_args()


def _resolve_scenario_dist(requested_episodes: int) -> dict:
    dist     = SCENARIO_DIST.copy()
    dist_sum = sum(dist.values())
    if requested_episodes == dist_sum:
        return dist
    ratio = requested_episodes / dist_sum
    dist  = {k: max(1, round(v * ratio)) for k, v in dist.items()}
    diff  = requested_episodes - sum(dist.values())
    if diff != 0:
        dist[max(dist, key=dist.get)] += diff
    return dist


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set.")

    input_file  = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        raise SystemExit(f"ERROR: {input_file} not found.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    total_episodes = args.episodes
    scenario_dist  = _resolve_scenario_dist(total_episodes)

    if sum(scenario_dist.values()) != total_episodes:
        raise SystemExit(
            f"ERROR: scenario_dist sums to {sum(scenario_dist.values())}, "
            f"expected {total_episodes}."
        )

    client = OpenAI(api_key=api_key)

    print("Loading SOP chunks...")
    index, hazard_index, all_chunks = load_chunks(input_file)
    mh_count = len(hazard_index.get("multi_hazard", []))
    print(
        f"  {len(all_chunks)} chunks | "
        f"{len(index)} topic groups | "
        f"{len(hazard_index)} hazard types | \n"
    )

    print(f"Scenario plan ({total_episodes} total episodes):")
    for sc, cnt in scenario_dist.items():
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<46} {cnt:>4} eps × {tmpl['turns']} turns")

    episodes_plan = plan_episodes(index, hazard_index, scenario_dist)

    print(f"\nGenerating (single-pass: generate + annotate)...\n")

    all_episodes: list[dict] = []
    failed = retried = 0

    for i, plan in enumerate(episodes_plan, start=1):
        sc   = plan["scenario"]
        tmpl = get_scenario_template(sc)
        n_exp = tmpl["turns"]

        print(
            f"[{i:>4}/{total_episodes}] {plan['episode_id']} | "
            f"{sc}:{tmpl['name'][:24]} | "
            f"{plan['primary_topic'][:36]}",
            end=" ", flush=True,
        )

        prompt = build_prompt(plan)

        try:
            ep = call_gpt(client, prompt)
        except Exception as e:
            print(f"✗ {type(e).__name__}: {e}")
            failed += 1
            continue

        meta_fields = [
            ("episode_id",        plan["episode_id"]),
            ("scenario_type",     sc),
            ("scenario_name",     tmpl["name"]),
            ("primary_topic",     plan["primary_topic"]),
            ("hazard_type",       plan["hazard_type"]),
            ("topic_group",       plan["topic_group"]),
            ("information_types", plan["information_types"]),
        ]
        for k, v in meta_fields:
            ep.setdefault(k, v)

        ok, reason = validate_episode(ep, n_exp)
        if not ok:
            print(f"⚠ ({reason}) retrying...", end=" ", flush=True)
            retried += 1
            try:
                ep = call_gpt(client, prompt)
                for k, v in meta_fields:
                    ep.setdefault(k, v)
                ok, reason = validate_episode(ep, n_exp)
            except Exception as e:
                ok, reason = False, str(e)

        if ok:
            all_episodes.append(ep)
            print("✓")
        else:
            print(f"✗ skipped ({reason})")
            failed += 1

        if i < total_episodes:
            time.sleep(SLEEP_BETWEEN)

    output = {
        "conversations": all_episodes,
        "metadata"     : build_metadata(all_episodes),
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    meta = output["metadata"]
    print(f"\n{'='*68}")
    print(
        f"Done!  saved={meta['total_episodes']}  "
        f"failed={failed}  retried={retried}"
    )
    print(f"Total turns : {meta['total_turns']}  |  avg/ep: {meta['avg_turns_per_episode']}")
    print(
        f"Retrieval required: {meta['retrieval_required_rate'] * 100:.1f}% "
        f"({meta['retrieval_true_turns']} true / {meta['retrieval_false_turns']} false)\n"
    )
    print("Scenario breakdown:")
    for sc in sorted(meta["scenario_distribution"]):
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<46} {meta['scenario_distribution'][sc]:>4} eps")
    print("\nQuery type breakdown:")
    for qt, cnt in sorted(meta["query_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {qt:<28} {cnt:>4} turns")
    print(f"\nOutput → {output_file}")


if __name__ == "__main__":
    main()
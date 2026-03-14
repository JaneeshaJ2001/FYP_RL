"""
Multi-Turn Disaster Dialogue Generator 
====================================================================
Generates conversation episodes grounded in chunks_grouped.json.
Supports 10 scenario templates (A–J).

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
    python generate_conversations_3.py
"""

import os
import json
import time
import random
from collections import defaultdict
from openai import OpenAI

from conversation_prompts import (
    DATASET_SYSTEM_PROMPT,
    VALID_QUERY_TYPES,
    RETRIEVAL_FALSE_TYPES,
    build_prompt,
    get_scenario_template,
)

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE     = "chunks_grouped.json"
OUTPUT_FILE    = "conversations.json"
MODEL          = "gpt-4o-mini"
MAX_TOKENS     = 3500
TOTAL_EPISODES = 500
CHUNKS_PER_EP  = 4          # chunks sampled per single-hazard episode
SLEEP_BETWEEN  = 0.5        # seconds between API calls
 
# ── Scenario distribution (must sum to TOTAL_EPISODES) ────────────────────────
SCENARIO_DIST = {
    "A": 60,   # Follow-up Conversation             6 turns
    "B": 60,   # Information Seeking                5 turns
    "C": 60,   # Emergency Dialogue                 6 turns
    "D": 50,   # Subtopic Shift Within Hazard        6 turns
    "E": 50,   # Ambiguous / Non-standalone          6 turns
    "F": 50,   # Unanswerable / Partial Answer       6 turns
    "G": 50,   # Lifecycle Progression               6 turns
    "H": 40,   # Comparative: Flood vs Landslide     6 turns
    "I": 40,   # Unsupported Scope / OOD             6 turns
    "J": 40,   # Misconception / Correction          5 turns
}
# Total = 500


# ── Load & index chunks ────────────────────────────────────────────────────────

def load_chunks(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = None
        for key in ("sop_chunks",):
            candidate = data.get(key)
            if isinstance(candidate, list):
                chunks = candidate
                break
        if chunks is None:
            raise ValueError(
                "Expected chunk input to be a list or an object "
                "containing a list under 'sop_chunks'."
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

def sample_chunks_for_episode(
    scenario: str,
    index: dict,
    hazard_index: dict,
    keys: list,
) -> tuple[list[dict], str, str]:
    """
    Scenario H uses chunks from both flood AND landslide.
    All other scenarios use a single hazard topic group.
    """
    if scenario == "H":
        flood_chunks     = hazard_index.get("flood", [])
        landslide_chunks = hazard_index.get("landslide", [])
        if not flood_chunks or not landslide_chunks:
            available = [h for h in hazard_index if hazard_index[h]]
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
            random.sample(flood_chunks,     min(2, len(flood_chunks))) +
            random.sample(landslide_chunks, min(2, len(landslide_chunks)))
        )
        return selected, hazard_label, "cross_hazard_comparison"

    key = random.choice(keys)
    pool = index[key]
    selected = random.sample(pool, min(CHUNKS_PER_EP, len(pool)))
    hazard, group = key
    return selected, hazard, group


# ── Episode planner ────────────────────────────────────────────────────────────

def plan_episodes(index: dict, hazard_index: dict, total: int) -> list:
    keys = list(index.keys())
    scenarios = []
    for sc, cnt in SCENARIO_DIST.items():
        scenarios.extend([sc] * cnt)
    random.shuffle(scenarios)
    scenarios = scenarios[:total]

    episodes = []
    for i, sc in enumerate(scenarios):
        selected, hazard, group = sample_chunks_for_episode(
            sc, index, hazard_index, keys
        )
        episodes.append({
            "episode_id"     : f"conv_{i+1:03d}",
            "scenario"       : sc,
            "primary_topic"  : f"{hazard} / {group}",
            "hazard_type"    : hazard,
            "topic_group"    : group,
            "selected_chunks": selected,
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
    "turn_id", "query", "assistant_answer", "relevant_chunks", "retrieval_required", "query_type",
}


def validate_episode(ep: dict, expected_turns: int) -> tuple[bool, str]:
    turns = ep.get("turns", [])
    if len(turns) != expected_turns:
        return False, f"expected {expected_turns} turns, got {len(turns)}"

    seen_chunk_ids: set[str] = set()   # tracks all chunks retrieved so far

    for t in turns:
        tid = t.get("turn_id", "?")

        # Required fields present
        missing = REQUIRED_TURN_FIELDS - t.keys()
        if missing:
            return False, f"turn {tid} missing fields: {missing}"

        # query_type valid
        qt = t["query_type"]
        if qt not in VALID_QUERY_TYPES:
            return False, f"turn {tid} bad query_type '{qt}'"

        # Types
        if not isinstance(t["retrieval_required"], bool):
            return False, f"turn {tid} retrieval_required is not bool"
        if not isinstance(t["relevant_chunks"], list):
            return False, f"turn {tid} relevant_chunks is not list"

        rr     = t["retrieval_required"]
        chunks = t["relevant_chunks"]

        # retrieval_required must be false for specific query types
        if qt in RETRIEVAL_FALSE_TYPES and rr:
            return False, (
                f"turn {tid} query_type='{qt}' requires retrieval_required=false"
            )

        # retrieval_required=true → must cite at least one chunk
        if rr and len(chunks) == 0:
            return False, (
                f"turn {tid} retrieval_required=true but relevant_chunks=[]"
            )

        # retrieval_required=false → relevant_chunks must be empty
        if not rr and len(chunks) > 0:
            return False, (
                f"turn {tid} retrieval_required=false but relevant_chunks is non-empty"
            )

        # retrieval_required=true → chunks must not have all been retrieved before
        if rr:
            new_chunks = set(chunks) - seen_chunk_ids
            if not new_chunks:
                return False, (
                    f"turn {tid} retrieval_required=true but all cited chunks "
                    f"were already retrieved in earlier turns"
                )
            seen_chunk_ids.update(chunks)
        # retrieval_required=false with followup_reuse/clarification:
        # chunks should have been seen before (soft check — warn, don't reject)

    return True, "ok"


# ── Metadata ───────────────────────────────────────────────────────────────────

def build_metadata(episodes: list) -> dict:
    sc_dist  = defaultdict(int)
    qt_dist  = defaultdict(int)
    top_dist = defaultdict(int)
    ret_t = total_t = 0

    for ep in episodes:
        sc_dist[ep.get("scenario_type", "?")] += 1
        top_dist[ep.get("primary_topic", "?")] += 1
        for t in ep.get("turns", []):
            total_t += 1
            qt_dist[t["query_type"]] += 1
            if t.get("retrieval_required"):
                ret_t += 1

    return {
        "total_episodes"          : len(episodes),
        "total_turns"             : total_t,
        "avg_turns_per_episode"   : round(total_t / len(episodes), 2) if episodes else 0,
        "retrieval_required_rate" : round(ret_t / total_t, 3) if total_t else 0,
        "scenario_distribution"   : dict(sc_dist),
        "topic_distribution"      : dict(top_dist),
        "query_type_distribution" : dict(qt_dist),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY not set.")
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"ERROR: {INPUT_FILE} not found.")
    if sum(SCENARIO_DIST.values()) != TOTAL_EPISODES:
        raise SystemExit(
            f"ERROR: SCENARIO_DIST sums to {sum(SCENARIO_DIST.values())}, "
            f"expected {TOTAL_EPISODES}."
        )

    client = OpenAI(api_key=api_key)

    print("Loading SOP chunks...")
    index, hazard_index, all_chunks = load_chunks(INPUT_FILE)
    print(
        f"  {len(all_chunks)} chunks | "
        f"{len(index)} topic groups | "
        f"{len(hazard_index)} hazard types\n"
    )

    print(f"Scenario plan ({TOTAL_EPISODES} total episodes):")
    for sc, cnt in SCENARIO_DIST.items():
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<42} {cnt:>3} eps × {tmpl['turns']} turns")

    episodes_plan = plan_episodes(index, hazard_index, TOTAL_EPISODES)
    print(f"\nGenerating (single-pass: generate + annotate)...\n")

    all_episodes = []
    failed = retried = 0

    for i, plan in enumerate(episodes_plan, start=1):
        sc    = plan["scenario"]
        tmpl  = get_scenario_template(sc)
        n_exp = tmpl["turns"]

        label = (
            f"[{i:>3}/{TOTAL_EPISODES}] {plan['episode_id']} | "
            f"{sc}:{tmpl['name'][:22]} | "
            f"{plan['primary_topic'][:35]}"
        )
        print(label, end=" ", flush=True)

        prompt = build_prompt(plan)

        try:
            ep = call_gpt(client, prompt)
        except Exception as e:
            print(f"✗ {type(e).__name__}: {e}")
            failed += 1
            continue

        # Inject metadata
        for k, v in [
            ("episode_id",    plan["episode_id"]),
            ("scenario_type", sc),
            ("scenario_name", tmpl["name"]),
            ("primary_topic", plan["primary_topic"]),
            ("hazard_type",   plan["hazard_type"]),
            ("topic_group",   plan["topic_group"]),
        ]:
            ep.setdefault(k, v)

        ok, reason = validate_episode(ep, n_exp)
        if not ok:
            print(f"⚠ ({reason}) retrying...", end=" ", flush=True)
            retried += 1
            try:
                ep = call_gpt(client, prompt)
                for k, v in [
                    ("episode_id",    plan["episode_id"]),
                    ("scenario_type", sc),
                    ("scenario_name", tmpl["name"]),
                    ("primary_topic", plan["primary_topic"]),
                    ("hazard_type",   plan["hazard_type"]),
                    ("topic_group",   plan["topic_group"]),
                ]:
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

        if i < TOTAL_EPISODES:
            time.sleep(SLEEP_BETWEEN)

    output = {
        "conversations": all_episodes,
        "metadata"     : build_metadata(all_episodes),
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    meta = output["metadata"]
    print(f"\n{'='*64}")
    print(
        f"Done!  saved={meta['total_episodes']}  "
        f"failed={failed}  retried={retried}"
    )
    print(f"Total turns : {meta['total_turns']}  |  avg/ep: {meta['avg_turns_per_episode']}")
    print(
        f"Retrieval rate: {meta['retrieval_required_rate']*100:.1f}% "
        f"of turns require new retrieval\n"
    )
    print("Scenario breakdown:")
    for sc in sorted(meta["scenario_distribution"]):
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<42} {meta['scenario_distribution'][sc]:>3} eps")
    print("\nQuery type breakdown:")
    for qt, cnt in sorted(
        meta["query_type_distribution"].items(), key=lambda x: -x[1]
    ):
        print(f"  {qt:<18} {cnt:>4} turns")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
"""
Multi-Turn Disaster Dialogue Generator  (two-step pipeline)
================================================================
Generates conversation episodes grounded in chunks_grouped.json.
Supports 10 scenario templates (A–J).

Pipeline per episode
────────────────────
  Step 1 — generate_conversation:  raw turns (no retrieval labels)
  Step 2 — label_turns:            assign retrieval_required + relevant_chunks

Scope
─────
• Floods and landslides ONLY.
• Single-hazard episodes (Scenario H is the sole cross-hazard comparison).
• OOD turns = unsupported disaster type OR unsupported local scope.

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
    LABELER_SYSTEM_PROMPT,
    build_generation_prompt,
    build_labeling_prompt,
    merge_labels,
    get_scenario_template,
)

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE     = "chunks_grouped.json"
OUTPUT_FILE    = "conversations.json"
MODEL          = "gpt-4o-mini"        # or "gpt-4o"
MAX_TOKENS_GEN = 3500                 # generation step
MAX_TOKENS_LBL = 1500                 # labeling step (much shorter output)
TOTAL_EPISODES = 10
CHUNKS_PER_EP  = 3                    # chunks sampled per single-hazard episode
SLEEP_BETWEEN  = 0.5                  # seconds between API calls

# ── Scenario distribution (must sum to TOTAL_EPISODES) ────────────────────────
#
#   A  Follow-up Conversation             6 turns
#   B  Information Seeking                5 turns
#   C  Emergency Dialogue                 6 turns
#   D  Subtopic Shift Within Hazard       6 turns
#   E  Ambiguous / Non-standalone         6 turns
#   F  Unanswerable / Partial Answer      6 turns
#   G  Lifecycle Progression              7 turns
#   H  Comparative (flood vs landslide)   6 turns
#   I  Unsupported Scope / OOD            6 turns
#   J  Misconception / Correction         5 turns

# SCENARIO_DIST = {
#     "A": 60,
#     "B": 60,
#     "C": 60,
#     "D": 50,
#     "E": 50,
#     "F": 50,
#     "G": 50,
#     "H": 40,
#     "I": 40,
#     "J": 40,
# }
SCENARIO_DIST = {
    "A": 1,
    "B": 1,
    "C": 1,
    "D": 1,
    "E": 1,
    "F": 1,
    "G": 1,
    "H": 1,
    "I": 1,
    "J": 1,
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
                "Expected chunk input JSON to be a list or an object containing "
                "a list under 'sop_chunks'."
            )
    else:
        raise ValueError("Expected chunk input JSON to be a list or object.")

    # Index by (hazard_type, topic_group) and by hazard_type alone
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
    Sample chunks for an episode.

    Scenario H (Comparative) uses chunks from BOTH flood and landslide.
    All other scenarios use a single hazard topic group.

    Returns (selected_chunks, hazard_label, group_label).
    """
    if scenario == "H":
        # Cross-hazard comparison — intentionally uses both supported hazards
        flood_chunks      = hazard_index.get("flood", [])
        landslide_chunks  = hazard_index.get("landslide", [])
        # Fallback: pick two different hazards if exact names differ
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

    # All other scenarios: single hazard, single topic group
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


# ── API helpers ────────────────────────────────────────────────────────────────

def call_gpt(client: OpenAI, prompt: str, system: str, max_tokens: int) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
    return json.loads(r.choices[0].message.content.strip())


# ── Validators ─────────────────────────────────────────────────────────────────

VALID_QT       = {"first_turn", "followup", "topic_shift", "clarification", "OOD", "unanswerable"}
TURN_FIELDS_GEN = {"turn_id", "query", "assistant_answer", "topic", "query_type"}
TURN_FIELDS_LBL = {"turn_id", "retrieval_required", "relevant_chunks"}


def validate_generation(ep: dict, expected_turns: int) -> tuple[bool, str]:
    turns = ep.get("turns", [])
    if len(turns) != expected_turns:
        return False, f"expected {expected_turns} turns, got {len(turns)}"
    for t in turns:
        missing = TURN_FIELDS_GEN - t.keys()
        if missing:
            return False, f"turn {t.get('turn_id', '?')} missing fields: {missing}"
        if t["query_type"] not in VALID_QT:
            return False, f"bad query_type '{t['query_type']}' in turn {t.get('turn_id', '?')}"
    return True, "ok"


def validate_labels(labels_resp: dict, expected_turns: int) -> tuple[bool, str]:
    labels = labels_resp.get("labels", [])
    if len(labels) != expected_turns:
        return False, f"expected {expected_turns} label entries, got {len(labels)}"
    for lb in labels:
        missing = TURN_FIELDS_LBL - lb.keys()
        if missing:
            return False, f"label turn {lb.get('turn_id', '?')} missing: {missing}"
        if not isinstance(lb["retrieval_required"], bool):
            return False, f"retrieval_required not bool in turn {lb.get('turn_id', '?')}"
        if not isinstance(lb["relevant_chunks"], list):
            return False, f"relevant_chunks not list in turn {lb.get('turn_id', '?')}"
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
        raise SystemExit(f"ERROR: {INPUT_FILE} not found. Run chunks_generate.py first.")
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
        print(f"  {sc}  {tmpl['name']:<40} {cnt:>3} eps × {tmpl['turns']} turns")

    episodes_plan = plan_episodes(index, hazard_index, TOTAL_EPISODES)
    print(f"\nGenerating (2-step pipeline: generate → label)...\n")

    all_episodes = []
    failed = retried_gen = retried_lbl = 0

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

        # ── Step 1: generate conversation ──────────────────────────────────────
        gen_prompt = build_generation_prompt(plan)
        try:
            ep = call_gpt(client, gen_prompt, DATASET_SYSTEM_PROMPT, MAX_TOKENS_GEN)
        except Exception as e:
            print(f"✗gen {type(e).__name__}: {e}")
            failed += 1
            continue

        # Inject metadata fields
        for k, v in [
            ("episode_id",    plan["episode_id"]),
            ("scenario_type", sc),
            ("scenario_name", tmpl["name"]),
            ("primary_topic", plan["primary_topic"]),
            ("hazard_type",   plan["hazard_type"]),
            ("topic_group",   plan["topic_group"]),
        ]:
            ep.setdefault(k, v)

        ok_gen, reason_gen = validate_generation(ep, n_exp)
        if not ok_gen:
            print(f"⚠gen({reason_gen}) retry...", end=" ", flush=True)
            retried_gen += 1
            try:
                ep = call_gpt(client, gen_prompt, DATASET_SYSTEM_PROMPT, MAX_TOKENS_GEN)
                for k, v in [
                    ("episode_id",    plan["episode_id"]),
                    ("scenario_type", sc),
                    ("scenario_name", tmpl["name"]),
                    ("primary_topic", plan["primary_topic"]),
                    ("hazard_type",   plan["hazard_type"]),
                    ("topic_group",   plan["topic_group"]),
                ]:
                    ep.setdefault(k, v)
                ok_gen, reason_gen = validate_generation(ep, n_exp)
            except Exception as e:
                ok_gen, reason_gen = False, str(e)

        if not ok_gen:
            print(f"✗gen skipped ({reason_gen})")
            failed += 1
            continue

        # ── Step 2: label turns ────────────────────────────────────────────────
        lbl_prompt = build_labeling_prompt(ep, plan["selected_chunks"])
        try:
            lbl_resp = call_gpt(client, lbl_prompt, LABELER_SYSTEM_PROMPT, MAX_TOKENS_LBL)
        except Exception as e:
            print(f"✗lbl {type(e).__name__}: {e}")
            failed += 1
            continue

        ok_lbl, reason_lbl = validate_labels(lbl_resp, n_exp)
        if not ok_lbl:
            print(f"⚠lbl({reason_lbl}) retry...", end=" ", flush=True)
            retried_lbl += 1
            try:
                lbl_resp = call_gpt(client, lbl_prompt, LABELER_SYSTEM_PROMPT, MAX_TOKENS_LBL)
                ok_lbl, reason_lbl = validate_labels(lbl_resp, n_exp)
            except Exception as e:
                ok_lbl, reason_lbl = False, str(e)

        if not ok_lbl:
            print(f"✗lbl skipped ({reason_lbl})")
            failed += 1
            continue

        # ── Merge and save ─────────────────────────────────────────────────────
        final_ep = merge_labels(ep, lbl_resp["labels"])
        all_episodes.append(final_ep)
        print("✓")

        if i < TOTAL_EPISODES:
            time.sleep(SLEEP_BETWEEN)

    # ── Write output ───────────────────────────────────────────────────────────
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
        f"failed={failed}  "
        f"retried_gen={retried_gen}  retried_lbl={retried_lbl}"
    )
    print(f"Total turns : {meta['total_turns']}  |  avg/ep: {meta['avg_turns_per_episode']}")
    print(f"Retrieval rate: {meta['retrieval_required_rate']*100:.1f}% of turns require retrieval\n")

    print("Scenario breakdown:")
    for sc in sorted(meta["scenario_distribution"]):
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<40} {meta['scenario_distribution'][sc]:>3} eps")

    print("\nQuery type breakdown:")
    for qt, cnt in sorted(meta["query_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {qt:<15} {cnt:>4} turns")

    print(f"\nOutput written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
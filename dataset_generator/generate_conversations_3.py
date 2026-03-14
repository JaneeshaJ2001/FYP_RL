"""
Multi-Turn Disaster Dialogue Generator
Generates 500 conversation episodes grounded in chunks_grouped.json.
Supports 10 scenario templates (A–J).

Usage:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    python generate_conversations.py

Input:  chunks_grouped.json   (produced by chunks_generate.py)
Output: conversations.json
"""

import os
import json
import time
import random
from collections import defaultdict
from openai import OpenAI

from conversation_prompts import DATASET_SYSTEM_PROMPT, build_prompt, get_scenario_template

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE     = "chunks_grouped.json"
OUTPUT_FILE    = "conversations.json"
MODEL          = "gpt-4o-mini"       # or "gpt-4o"
MAX_TOKENS     = 3500
TOTAL_EPISODES = 500
CHUNKS_PER_EP  = 3              # chunks sampled per episode for single-topic scenarios
SLEEP_BETWEEN  = 0.5            # seconds between API calls

# ── Scenario distribution (must sum to TOTAL_EPISODES = 500) ──────────────────
#
#   A  Follow-up Conversation             6 turns
#   B  Information Seeking                5 turns
#   C  Emergency Dialogue                 6 turns
#   D  Topic-Shift                        6 turns
#   E  Ambiguous / Non-standalone         6 turns
#   F  Unanswerable / Partial Answer      6 turns
#   G  Cross-Stage Progression            7 turns
#   H  Cross-Hazard Comparison            6 turns
#   I  OOD / Irrelevant Query             6 turns
#   J  Contradiction / Correction         5 turns

SCENARIO_DIST = {
    "A": 60,
    "B": 60,
    "C": 60,
    "D": 50,
    "E": 50,
    "F": 50,
    "G": 50,
    "H": 40,
    "I": 40,
    "J": 40,
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
        for key in ("sop_chunks"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                chunks = candidate
                break
        if chunks is None:
            raise ValueError(
                "Expected chunk input JSON to be a list or an object containing a list under sop_chunks"
            )
    else:
        raise ValueError("Expected chunk input JSON to be a list or object")

    index        = defaultdict(list)   # (hazard, group) → chunks
    hazard_index = defaultdict(list)   # hazard → chunks

    for c in chunks:
        hazard = c.get("hazard_type") or c.get("topic", "general").split()[0]
        group  = c.get("topic_group") or c.get("topic", "general")
        index[(hazard, group)].append(c)
        hazard_index[hazard].append(c)

    return index, hazard_index, chunks


# ── Chunk sampler ──────────────────────────────────────────────────────────────

def sample_chunks(sc: str, index: dict, hazard_index: dict, keys: list):
    mode = get_scenario_template(sc)["chunks_mode"]
    if mode == "multi_topic":
        available = list(hazard_index.keys())
        if len(available) >= 2:
            h1, h2 = random.sample(available, 2)
        else:
            h1 = h2 = available[0]
        pool1    = hazard_index[h1]
        pool2    = hazard_index[h2]
        selected = random.sample(pool1, min(2, len(pool1))) + random.sample(pool2, min(2, len(pool2)))
        hazard, group = f"{h1}+{h2}", "multi_topic"
    else:
        key      = random.choice(keys)
        pool     = index[key]
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
        selected, hazard, group = sample_chunks(sc, index, hazard_index, keys)
        episodes.append({
            "episode_id"     : f"conv_{i+1:03d}",
            "scenario"       : sc,
            "primary_topic"  : f"{hazard} / {group}",
            "hazard_type"    : hazard,
            "topic_group"    : group,
            "selected_chunks": selected,
        })
    return episodes


# ── API caller ─────────────────────────────────────────────────────────────────

def call_gpt(client: OpenAI, prompt: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DATASET_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return json.loads(r.choices[0].message.content.strip())


# ── Validator ──────────────────────────────────────────────────────────────────

VALID_QT    = {"first_turn", "followup", "topic_shift", "clarification", "OOD", "unanswerable"}
TURN_FIELDS = {"turn_id", "query", "assistant_answer", "relevant_chunks", "retrieval_required", "topic", "query_type"}

def validate_episode(ep: dict, expected_turns: int) -> tuple:
    turns = ep.get("turns", [])
    if len(turns) != expected_turns:
        return False, f"expected {expected_turns} turns, got {len(turns)}"
    for t in turns:
        missing = TURN_FIELDS - t.keys()
        if missing:
            return False, f"turn {t.get('turn_id','?')} missing: {missing}"
        if t["query_type"] not in VALID_QT:
            return False, f"bad query_type '{t['query_type']}' in turn {t.get('turn_id','?')}"
        if not isinstance(t["retrieval_required"], bool):
            return False, f"retrieval_required not bool in turn {t.get('turn_id','?')}"
        if not isinstance(t["relevant_chunks"], list):
            return False, f"relevant_chunks not list in turn {t.get('turn_id','?')}"
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
            if t["retrieval_required"]:
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
        raise SystemExit(f"ERROR: SCENARIO_DIST sums to {sum(SCENARIO_DIST.values())}, expected {TOTAL_EPISODES}.")

    client = OpenAI(api_key=api_key)

    print("Loading SOP chunks...")
    index, hazard_index, all_chunks = load_chunks(INPUT_FILE)
    print(f"  {len(all_chunks)} chunks | {len(index)} topic groups | {len(hazard_index)} hazard types\n")

    print(f"Scenario plan ({TOTAL_EPISODES} total episodes):")
    for sc, cnt in SCENARIO_DIST.items():
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<32} {cnt:>3} eps × {tmpl['turns']} turns")

    episodes_plan = plan_episodes(index, hazard_index, TOTAL_EPISODES)
    print(f"\nGenerating...\n")

    all_episodes = []
    failed = retried = 0

    for i, plan in enumerate(episodes_plan, start=1):
        sc     = plan["scenario"]
        tmpl   = get_scenario_template(sc)
        n_exp  = tmpl["turns"]
        prompt = build_prompt(plan)

        label = (f"[{i:>3}/{TOTAL_EPISODES}] {plan['episode_id']} | "
                 f"{sc}:{tmpl['name'][:20]} | "
                 f"{plan['primary_topic'][:35]}")
        print(label, end=" ... ", flush=True)

        try:
            ep = call_gpt(client, prompt)
        except (json.JSONDecodeError, Exception) as e:
            print(f"✗ {type(e).__name__}: {e}")
            failed += 1
            continue

        for k, v in [("episode_id", plan["episode_id"]), ("scenario_type", sc),
                 ("scenario_name", tmpl["name"]),
                     ("primary_topic", plan["primary_topic"]),
                     ("hazard_type", plan["hazard_type"]), ("topic_group", plan["topic_group"])]:
            ep.setdefault(k, v)

        ok, reason = validate_episode(ep, n_exp)
        if not ok:
            print(f"⚠ ({reason}) retrying...", end=" ", flush=True)
            retried += 1
            try:
                ep = call_gpt(client, prompt)
                for k, v in [("episode_id", plan["episode_id"]), ("scenario_type", sc),
                              ("scenario_name", tmpl["name"]),
                              ("primary_topic", plan["primary_topic"]),
                              ("hazard_type", plan["hazard_type"]), ("topic_group", plan["topic_group"])]:
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

    output = {"conversations": all_episodes, "metadata": build_metadata(all_episodes)}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    meta = output["metadata"]
    print(f"\n{'='*62}")
    print(f"Done!  saved={meta['total_episodes']}  failed={failed}  retried={retried}")
    print(f"Total turns : {meta['total_turns']}  |  avg/ep: {meta['avg_turns_per_episode']}")
    print(f"Retrieval rate: {meta['retrieval_required_rate']*100:.1f}% of turns require retrieval\n")
    print("Scenario breakdown:")
    for sc in sorted(meta["scenario_distribution"]):
        tmpl = get_scenario_template(sc)
        print(f"  {sc}  {tmpl['name']:<32} {meta['scenario_distribution'][sc]:>3} eps")
    print("\nQuery type breakdown:")
    for qt, cnt in sorted(meta["query_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {qt:<15} {cnt:>4} turns")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
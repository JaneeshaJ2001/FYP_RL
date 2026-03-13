"""
Multi-Turn Disaster Dialogue Generator
Generates 200 conversation episodes grounded in chunks_grouped.json.
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

# ── Configuration ──────────────────────────────────────────────────────────────

INPUT_FILE     = "chunks_grouped.json"
OUTPUT_FILE    = "conversations.json"
MODEL          = "gpt-4o-mini"       # or "gpt-4o"
MAX_TOKENS     = 3500
TOTAL_EPISODES = 200
CHUNKS_PER_EP  = 3              # chunks sampled per episode for single-topic scenarios
SLEEP_BETWEEN  = 0.5            # seconds between API calls

# ── Scenario distribution (must sum to TOTAL_EPISODES = 200) ──────────────────
#
#   A  Follow-up Conversation          (original)   6 turns
#   B  Information Seeking             (original)   5 turns
#   C  Emergency Dialogue              (original)   6 turns
#   D  Topic-Shift                     (new)        6 turns
#   E  Ambiguous / Non-standalone      (new)        6 turns
#   F  Unanswerable / Partial Answer   (new)        6 turns
#   G  Cross-Stage Progression         (new)        7 turns
#   H  Cross-Hazard Comparison         (new)        6 turns
#   I  OOD / Irrelevant Query          (new)        6 turns
#   J  Contradiction / Correction      (new)        5 turns

SCENARIO_DIST = {
    "A": 25,
    "B": 25,
    "C": 25,
    "D": 20,
    "E": 20,
    "F": 20,
    "G": 20,
    "H": 15,
    "I": 15,
    "J": 15,
}
# Total = 200

# ── Scenario templates ─────────────────────────────────────────────────────────

SCENARIO_TEMPLATES = {

    "A": {
        "name": "Follow-up Conversation",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: first question about the main topic (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: follow-up detail question answerable from Turn 1 (retrieval_required=false, query_type=followup)\n"
            "Turn 3: another follow-up, still on same topic (retrieval_required=false, query_type=followup)\n"
            "Turn 4: topic shift to a related but different hazard/procedure (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 5: clarification about something ambiguous in Turn 4 answer (retrieval_required=true, query_type=clarification)\n"
            "Turn 6: question completely outside disaster response KB (retrieval_required=false, query_type=OOD)"
        ),
        "extra_instructions": "",
    },

    "B": {
        "name": "Information Seeking",
        "turns": 5,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: direct factual query about the main topic (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: request for more detail on one specific point (retrieval_required=true, query_type=followup)\n"
            "Turn 3: 'why' or reasoning question — answerable from prior context (retrieval_required=false, query_type=followup)\n"
            "Turn 4: vague or ambiguous question needing KB lookup (retrieval_required=true, query_type=clarification)\n"
            "Turn 5: question the KB cannot answer at all (retrieval_required=false, query_type=unanswerable)"
        ),
        "extra_instructions": "",
    },

    "C": {
        "name": "Emergency Dialogue",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: urgent question from someone in an active emergency (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: safety follow-up answerable from Turn 1 context (retrieval_required=false, query_type=followup)\n"
            "Turn 3: clarification about a specific safety step (retrieval_required=true, query_type=clarification)\n"
            "Turn 4: topic shift — user mentions a second nearby hazard (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 5: rescue or evacuation advice for the second hazard (retrieval_required=true, query_type=followup)\n"
            "Turn 6: recovery question after the immediate emergency (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": "",
    },

    "D": {
        "name": "Topic-Shift",
        "turns": 6,
        "chunks_mode": "multi_topic",
        "description": (
            "Turn 1: question about FLOOD preparedness (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: follow-up on the flood answer (retrieval_required=false, query_type=followup)\n"
            "Turn 3: sudden switch — user asks about LANDSLIDE warning signs (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 4: follow-up on landslide response from Turn 3 (retrieval_required=false, query_type=followup)\n"
            "Turn 5: switch again to SHELTER, MEDICAL, or COMMUNICATION procedures (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 6: follow-up on shelter/medical/communication from Turn 5 (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Use chunks from at least two different hazard types. "
            "The topic field in each turn must accurately reflect the current hazard being discussed — "
            "not always the primary_topic. Adapt chunk selection across turns for realism."
        ),
    },

    "E": {
        "name": "Ambiguous / Non-Standalone Query",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: detailed, specific question about the main disaster topic (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: user follow-up 'What about children?' — ambiguous, references Turn 1 context (retrieval_required=true, query_type=clarification)\n"
            "Turn 3: user asks 'Is it safe?' — completely non-standalone, no explicit subject (retrieval_required=false, query_type=clarification)\n"
            "Turn 4: user asks 'How long should we wait?' — non-standalone, implicit reference (retrieval_required=false, query_type=clarification)\n"
            "Turn 5: user asks a slightly broader related question needing KB (retrieval_required=true, query_type=followup)\n"
            "Turn 6: user asks something unanswerable — specific local detail not in SOP (retrieval_required=false, query_type=unanswerable)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turns 3 and 4 must have queries that make NO sense without prior context. "
            "The assistant must resolve pronouns ('it', 'they', 'we', 'children') from conversation memory, "
            "NOT from new KB lookups. relevant_chunks=[] for Turns 3 and 4."
        ),
    },

    "F": {
        "name": "Unanswerable / Partial Answer",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: question clearly answerable from KB chunks (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: asks for specific LOCAL detail not in SOP (e.g. 'Which exact shelter in Colombo District?') (retrieval_required=false, query_type=unanswerable)\n"
            "Turn 3: asks for EXACT timing SOP does not specify (e.g. 'Exactly how many hours before the flood?') (retrieval_required=false, query_type=unanswerable)\n"
            "Turn 4: asks for a hotline for an UNKNOWN district not listed in KB (retrieval_required=false, query_type=unanswerable)\n"
            "Turn 5: question PARTIALLY supported — KB gives general info but not the specific detail asked (retrieval_required=true, query_type=followup) — assistant answers what KB supports and acknowledges the gap\n"
            "Turn 6: user rephrases Turn 4 with broader scope — now answerable from KB (retrieval_required=true, query_type=clarification)"
        ),
        "extra_instructions": (
            "IMPORTANT: For unanswerable turns, the assistant_answer must explicitly say the KB does not contain "
            "this specific information and suggest contacting local authorities. "
            "Do NOT invent hotlines, specific locations, or exact timelines."
        ),
    },

    "G": {
        "name": "Cross-Stage Progression",
        "turns": 7,
        "chunks_mode": "multi_topic",
        "description": (
            "Turn 1: PREPAREDNESS — what to do before disaster strikes (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: WARNING/MONITORING — what signs indicate the disaster is approaching (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 3: follow-up on warning signs from Turn 2 (retrieval_required=false, query_type=followup)\n"
            "Turn 4: EVACUATION — when and how to evacuate (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 5: RESCUE — what to do if someone is trapped or injured (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 6: RECOVERY — how to safely return and recover after the event (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 7: follow-up on recovery from Turn 6 (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: This episode must progress through disaster lifecycle stages: "
            "preparedness → warning/monitoring → evacuation → rescue → recovery. "
            "Each stage change is a topic_shift. Use different chunks for different stages where possible. "
            "The topic field per turn must reflect the CURRENT STAGE, "
            "e.g. 'flood / preparedness', 'flood / evacuation', 'flood / rescue_operations'."
        ),
    },

    "H": {
        "name": "Cross-Hazard Comparison",
        "turns": 6,
        "chunks_mode": "multi_topic",
        "description": (
            "Turn 1: ask about FLOOD warning signs (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: ask how FLOOD evacuation works (retrieval_required=true, query_type=followup)\n"
            "Turn 3: compare — ask about LANDSLIDE warning signs and how they differ from flood (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 4: compare safe SHELTER behaviour for flood vs landslide (retrieval_required=true, query_type=clarification)\n"
            "Turn 5: ask which hazard requires FASTER evacuation and why — reason from prior turns only (retrieval_required=false, query_type=followup)\n"
            "Turn 6: practical comparison of supply kits for both hazards (retrieval_required=true, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Use chunks from BOTH flood AND landslide topic groups. "
            "Turn 5 is a comparative reasoning turn — assistant must synthesise from conversation history only. "
            "relevant_chunks=[] for Turn 5."
        ),
    },

    "I": {
        "name": "OOD / Irrelevant Query",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: normal on-topic disaster question (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: follow-up on disaster topic (retrieval_required=false, query_type=followup)\n"
            "Turn 3: another relevant disaster question (retrieval_required=true, query_type=followup)\n"
            "Turn 4: OOD — user asks something completely unrelated to disaster response (e.g. recipe, sports, travel) (retrieval_required=false, query_type=OOD)\n"
            "Turn 5: OOD — user continues the off-topic tangent (retrieval_required=false, query_type=OOD)\n"
            "Turn 6: user returns to disaster topic (retrieval_required=true, query_type=first_turn)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turns 4 and 5 must be genuinely unrelated to disaster response. "
            "The assistant should briefly acknowledge but NOT retrieve from the disaster KB. "
            "relevant_chunks=[] for Turns 4 and 5. "
            "The assistant may politely note it is a disaster-response assistant."
        ),
    },

    "J": {
        "name": "Contradiction / Correction",
        "turns": 5,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: user states or assumes an UNSAFE or INCORRECT action and asks to confirm it "
                    "(e.g. 'Is it okay to stay home during a flood if I live on the second floor?') "
                    "(retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: assistant CORRECTS the unsafe assumption firmly, grounded in KB (retrieval_required=true, query_type=clarification)\n"
            "Turn 3: user asks a follow-up BASED ON the correction "
                    "(e.g. 'If I shouldn't stay, when exactly should I leave?') (retrieval_required=true, query_type=followup)\n"
            "Turn 4: user makes ANOTHER incorrect assumption related to the topic (retrieval_required=true, query_type=clarification)\n"
            "Turn 5: assistant corrects again with KB grounding; user accepts and asks a safe follow-up (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turns 1 and 4 must start with user assumptions that are factually WRONG or DANGEROUS "
            "according to the SOP. The assistant MUST NEVER validate the unsafe assumption. "
            "Corrections must be grounded in the provided chunks with relevant_chunks populated."
        ),
    },
}

# ── Load & index chunks ────────────────────────────────────────────────────────

def load_chunks(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = None
        for key in ("chunks_grouped", "chunks", "sop_chunks", "data"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                chunks = candidate
                break
        if chunks is None:
            raise ValueError(
                "Expected chunk input JSON to be a list or an object containing a list under one of: chunks_grouped, chunks, sop_chunks, data"
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
    mode = SCENARIO_TEMPLATES[sc]["chunks_mode"]
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


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_prompt(ep: dict) -> str:
    sc      = ep["scenario"]
    tmpl    = SCENARIO_TEMPLATES[sc]
    n       = tmpl["turns"]
    topic   = ep["primary_topic"]

    chunk_block = "\n\n".join(
        f"[{c['chunk_id']}] hazard={c.get('hazard_type','?')} | stage={c.get('topic_group','?')}\n{c['text']}"
        for c in ep["selected_chunks"]
    )
    chunk_ids = [c["chunk_id"] for c in ep["selected_chunks"]]
    extra     = f"\n\n## ADDITIONAL INSTRUCTIONS\n{tmpl['extra_instructions']}" if tmpl["extra_instructions"] else ""

    return f"""You are generating a disaster-response conversation dataset.
The dataset trains a PPO policy to decide WHEN to retrieve from a knowledge base vs use conversation memory.

## SCENARIO: {sc} — {tmpl['name']}
## PRIMARY TOPIC: {topic}
## AVAILABLE CHUNK IDs: {chunk_ids}

## KNOWLEDGE BASE (use ONLY these chunks for retrieval_required=true turns)
{chunk_block}

## TURN-BY-TURN STRUCTURE — follow exactly
{tmpl['description']}
{extra}

## STRICT RULES
1. Resident speaks naturally — scared, confused, urgent, or curious depending on scenario.
2. Assistant is calm, concise (1–3 sentences), professional.
3. retrieval_required=true → assistant answer MUST be grounded in KB chunks above.
4. retrieval_required=false → assistant answers from conversation context ONLY, no new KB facts.
5. relevant_chunks: list of exact chunk_id strings used, or [] if not grounded in KB.
6. query_type: EXACTLY one of: first_turn | followup | topic_shift | clarification | OOD | unanswerable
7. retrieval_required logic:
   new topic / first factual need                 → true
   follow-up answerable from prior assistant turn → false
   ambiguous / clarification needing KB           → true
   OOD (unrelated to disaster KB)                 → false
   unanswerable (KB cannot answer)                → false
8. Every turn: turn_id, query, assistant_answer, relevant_chunks, retrieval_required, topic, query_type
9. topic field reflects the CURRENT subject of that turn (may change per turn).

## OUTPUT — ONLY valid JSON, no markdown:
{{
  "episode_id": "{ep['episode_id']}",
  "scenario_type": "{sc}",
  "scenario_name": "{tmpl['name']}",
  "primary_topic": "{topic}",
  "hazard_type": "{ep['hazard_type']}",
  "topic_group": "{ep['topic_group']}",
  "turns": [
    {{
      "turn_id": 1,
      "query": "...",
      "assistant_answer": "...",
      "relevant_chunks": ["SOP_XXX"],
      "retrieval_required": true,
      "topic": "...",
      "query_type": "first_turn"
    }}
  ]
}}

Generate the complete {n}-turn episode now. Ensure EXACTLY {n} turns."""


# ── API caller ─────────────────────────────────────────────────────────────────

def call_gpt(client: OpenAI, prompt: str) -> dict:
    r = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "You are an expert synthetic dataset generator for disaster-response RL research. "
                "Respond with valid JSON only. No markdown. No explanations."
            )},
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
        print(f"  {sc}  {SCENARIO_TEMPLATES[sc]['name']:<32} {cnt:>3} eps × {SCENARIO_TEMPLATES[sc]['turns']} turns")

    episodes_plan = plan_episodes(index, hazard_index, TOTAL_EPISODES)
    print(f"\nGenerating...\n")

    all_episodes = []
    failed = retried = 0

    for i, plan in enumerate(episodes_plan, start=1):
        sc     = plan["scenario"]
        n_exp  = SCENARIO_TEMPLATES[sc]["turns"]
        prompt = build_prompt(plan)

        label = (f"[{i:>3}/{TOTAL_EPISODES}] {plan['episode_id']} | "
                 f"{sc}:{SCENARIO_TEMPLATES[sc]['name'][:20]} | "
                 f"{plan['primary_topic'][:35]}")
        print(label, end=" ... ", flush=True)

        try:
            ep = call_gpt(client, prompt)
        except (json.JSONDecodeError, Exception) as e:
            print(f"✗ {type(e).__name__}: {e}")
            failed += 1
            continue

        for k, v in [("episode_id", plan["episode_id"]), ("scenario_type", sc),
                     ("scenario_name", SCENARIO_TEMPLATES[sc]["name"]),
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
                              ("scenario_name", SCENARIO_TEMPLATES[sc]["name"]),
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
        print(f"  {sc}  {SCENARIO_TEMPLATES[sc]['name']:<32} {meta['scenario_distribution'][sc]:>3} eps")
    print("\nQuery type breakdown:")
    for qt, cnt in sorted(meta["query_type_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {qt:<15} {cnt:>4} turns")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
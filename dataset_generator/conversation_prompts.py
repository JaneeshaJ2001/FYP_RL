"""Prompt and scenario definitions for conversation dataset generation.

Design principles
─────────────────
• All episodes are single-hazard (flood OR landslide).  No artificial multi-topic mixing.
• Scenario templates describe conversational INTENT PROGRESSION only.
  They never hardcode `retrieval_required` — that is determined in a separate
  labeling step after the conversation is generated.
• OOD turns are scoped to unsupported disaster types / unsupported local scope,
  NOT unrelated real-world topics (recipes, sports, etc.).
• Two-step pipeline:
    Step 1 – generate_conversation_prompt()  → raw turns (no retrieval labels)
    Step 2 – label_turns_prompt()            → per-turn retrieval_required + relevant_chunks
"""

from __future__ import annotations

from typing import Any, Dict


# ── System prompts ─────────────────────────────────────────────────────────────

DATASET_SYSTEM_PROMPT = (
    "You are an expert synthetic dataset generator for disaster-response RL research. "
    "Respond with valid JSON only. No markdown. No explanations."
)

LABELER_SYSTEM_PROMPT = (
    "You are a precise annotation engine for disaster-response conversation datasets. "
    "Respond with valid JSON only. No markdown. No explanations."
)


# ── Scenario templates ─────────────────────────────────────────────────────────
# Each template defines:
#   name        – human-readable label
#   turns       – expected turn count
#   description – intent progression per turn (NO retrieval labels)
#   extra_instructions – additional generation constraints
#
# chunks_mode is REMOVED.  All episodes are single-hazard.
# retrieval_required is REMOVED from descriptions — assigned by labeler.

SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Follow-up Conversation",
        "turns": 6,
        "description": (
            "Turn 1: user asks a first factual question about the main hazard topic. (query_type=first_turn)\n"
            "Turn 2: user asks a follow-up detail question directly related to Turn 1. (query_type=followup)\n"
            "Turn 3: user asks another follow-up, still on the same sub-topic. (query_type=followup)\n"
            "Turn 4: user shifts to a related but different procedure or stage within the SAME hazard. (query_type=topic_shift)\n"
            "Turn 5: user asks for clarification about something in the Turn 4 answer. (query_type=clarification)\n"
            "Turn 6: user asks about an unsupported scope — a specific local detail (shelter name, hotline) "
            "not available in the KB; assistant says it cannot provide that specific information "
            "and directs the user to local authorities. (query_type=unanswerable)"
        ),
        "extra_instructions": "",
    },
    "B": {
        "name": "Information Seeking",
        "turns": 5,
        "description": (
            "Turn 1: user asks a direct factual query about the main hazard topic. (query_type=first_turn)\n"
            "Turn 2: user asks for more detail on one specific point from Turn 1. (query_type=followup)\n"
            "Turn 3: user asks a 'why' or reasoning question answerable from prior assistant context. (query_type=followup)\n"
            "Turn 4: user asks a vague or ambiguous question that may need KB clarification. (query_type=clarification)\n"
            "Turn 5: user asks about an unsupported disaster type (e.g. earthquake, tsunami) — "
            "assistant replies it only covers floods and landslides. (query_type=OOD)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turn 5 must be about a real disaster but one NOT supported by this assistant "
            "(e.g. earthquake, cyclone, tsunami, drought). "
            "The assistant must clearly state it only covers floods and landslides."
        ),
    },
    "C": {
        "name": "Emergency Dialogue",
        "turns": 6,
        "description": (
            "Turn 1: user asks an urgent question from inside an active flood or landslide emergency. (query_type=first_turn)\n"
            "Turn 2: user asks a safety follow-up using context already given in Turn 1. (query_type=followup)\n"
            "Turn 3: user asks for clarification about a specific safety step mentioned in Turn 2. (query_type=clarification)\n"
            "Turn 4: user mentions a related hazard sub-topic within the SAME hazard type "
            "(e.g. contaminated water, structural damage). (query_type=topic_shift)\n"
            "Turn 5: user asks a rescue or evacuation question building on Turn 4. (query_type=followup)\n"
            "Turn 6: user asks a recovery or post-event question. (query_type=followup)"
        ),
        "extra_instructions": "",
    },
    "D": {
        "name": "Subtopic Shift Within Hazard",
        "turns": 6,
        "description": (
            "Turn 1: user asks about PREPAREDNESS for the hazard (what to do before it strikes). (query_type=first_turn)\n"
            "Turn 2: user asks a follow-up on the preparedness answer. (query_type=followup)\n"
            "Turn 3: user shifts to WARNING SIGNS — how to recognise the hazard is imminent. (query_type=topic_shift)\n"
            "Turn 4: user asks a follow-up on warning signs from Turn 3. (query_type=followup)\n"
            "Turn 5: user shifts to EVACUATION — when and how to leave. (query_type=topic_shift)\n"
            "Turn 6: user asks a follow-up on evacuation from Turn 5. (query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: All turns must stay within the SAME hazard type. "
            "The topic field per turn must reflect the current sub-stage, "
            "e.g. 'flood / preparedness', 'flood / warning_signs', 'flood / evacuation'."
        ),
    },
    "E": {
        "name": "Ambiguous / Non-Standalone Query",
        "turns": 6,
        "description": (
            "Turn 1: user asks a detailed, specific question about the main disaster topic. (query_type=first_turn)\n"
            "Turn 2: user asks an ambiguous follow-up that references Turn 1 context "
            "(e.g. 'What about children?' — no explicit subject). (query_type=clarification)\n"
            "Turn 3: user asks 'Is it safe now?' — completely non-standalone, no explicit subject. (query_type=clarification)\n"
            "Turn 4: user asks 'How long should we wait?' — non-standalone, implicit reference. (query_type=clarification)\n"
            "Turn 5: user asks a slightly broader related question that may require KB. (query_type=followup)\n"
            "Turn 6: user asks a location-specific detail not in the KB "
            "(e.g. exact shelter address); assistant says it cannot provide that. (query_type=unanswerable)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turns 3 and 4 must be queries that make NO sense without prior conversation context. "
            "The assistant must resolve pronouns and implicit references from conversation memory only — "
            "NOT from new KB lookups."
        ),
    },
    "F": {
        "name": "Unanswerable / Partial Answer",
        "turns": 6,
        "description": (
            "Turn 1: user asks a question clearly answerable from the KB chunks. (query_type=first_turn)\n"
            "Turn 2: user asks for a specific LOCAL detail not in the KB "
            "(e.g. 'Which exact shelter is open in my district?'). (query_type=unanswerable)\n"
            "Turn 3: user asks for an EXACT timing the KB does not specify "
            "(e.g. 'Exactly how many hours before the flood should I leave?'). (query_type=unanswerable)\n"
            "Turn 4: user asks for a hotline for a district not listed in the KB. (query_type=unanswerable)\n"
            "Turn 5: user asks a question PARTIALLY supported — KB gives general info but not the "
            "specific detail asked; assistant answers what KB supports and acknowledges the gap. (query_type=followup)\n"
            "Turn 6: user rephrases Turn 4 with a broader scope now answerable from the KB. (query_type=clarification)"
        ),
        "extra_instructions": (
            "IMPORTANT: For unanswerable turns, the assistant_answer must explicitly say the KB does not contain "
            "this specific information and suggest contacting local authorities. "
            "Do NOT invent hotlines, specific locations, or exact timelines."
        ),
    },
    "G": {
        "name": "Lifecycle Progression",
        "turns": 7,
        "description": (
            "Turn 1: PREPAREDNESS — user asks what to do before the disaster strikes. (query_type=first_turn)\n"
            "Turn 2: WARNING/MONITORING — user asks what signs indicate the disaster is approaching. (query_type=topic_shift)\n"
            "Turn 3: follow-up on warning signs from Turn 2. (query_type=followup)\n"
            "Turn 4: EVACUATION — user asks when and how to evacuate. (query_type=topic_shift)\n"
            "Turn 5: RESCUE — user asks what to do if someone is trapped or injured. (query_type=topic_shift)\n"
            "Turn 6: RECOVERY — user asks how to safely return and recover after the event. (query_type=topic_shift)\n"
            "Turn 7: follow-up on recovery from Turn 6. (query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: All turns must stay within the SAME hazard type. "
            "The episode must progress through: preparedness → warning/monitoring → evacuation → rescue → recovery. "
            "The topic field per turn must reflect the CURRENT STAGE, "
            "e.g. 'flood / preparedness', 'flood / evacuation', 'flood / rescue_operations'."
        ),
    },
    "H": {
        "name": "Comparative Within Supported Hazards",
        "turns": 6,
        "description": (
            "Turn 1: user asks about warning signs for FLOOD. (query_type=first_turn)\n"
            "Turn 2: user asks how flood evacuation works. (query_type=followup)\n"
            "Turn 3: user asks about LANDSLIDE warning signs and how they differ from flood. (query_type=topic_shift)\n"
            "Turn 4: user asks how shelter behaviour differs for flood vs landslide. (query_type=clarification)\n"
            "Turn 5: user asks which hazard requires faster evacuation and why — "
            "answerable by reasoning from prior assistant turns only, no new KB needed. (query_type=followup)\n"
            "Turn 6: user asks about practical supply kits and whether they differ between the two hazards. (query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: This is the ONE scenario that spans both flood AND landslide within a single episode — "
            "because the user is explicitly comparing the two supported hazards. "
            "Use chunks from both hazard types. "
            "Turn 5 is a comparative reasoning turn — assistant synthesises from conversation history only; "
            "no new chunk evidence is needed."
        ),
    },
    "I": {
        "name": "Unsupported Scope / Out-of-Domain",
        "turns": 6,
        "description": (
            "Turn 1: user asks a normal on-topic flood or landslide question. (query_type=first_turn)\n"
            "Turn 2: user asks a follow-up on the same topic. (query_type=followup)\n"
            "Turn 3: user asks another relevant in-scope question. (query_type=followup)\n"
            "Turn 4: user asks about an UNSUPPORTED disaster type such as earthquake, cyclone, tsunami, "
            "or wildfire — assistant says it only covers floods and landslides and cannot help with this. (query_type=OOD)\n"
            "Turn 5: user asks for a very specific LOCAL detail outside the KB "
            "(e.g. exact road closure, specific village officer contact) — "
            "assistant says it does not have that location-specific information. (query_type=unanswerable)\n"
            "Turn 6: user returns to a flood or landslide topic. (query_type=first_turn)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turn 4 must be about a real natural disaster that is NOT floods or landslides. "
            "The assistant must clearly state: 'I can only assist with floods and landslides.' "
            "Turn 5 must be a legitimate but hyper-local question (road name, specific official contact) "
            "that would not appear in any SOP. "
            "The assistant must acknowledge the limitation and direct the user to local authorities. "
            "Do NOT use unrelated non-disaster topics (recipes, sports, travel, etc.)."
        ),
    },
    "J": {
        "name": "Misconception / Correction",
        "turns": 5,
        "description": (
            "Turn 1: user states or assumes an UNSAFE or INCORRECT action and asks to confirm it "
            "(e.g. 'Is it okay to stay home during a flood if I live on the second floor?'). (query_type=first_turn)\n"
            "Turn 2: assistant CORRECTS the unsafe assumption firmly, grounded in KB. (query_type=clarification)\n"
            "Turn 3: user asks a follow-up BASED ON the correction "
            "(e.g. 'If I should not stay, when exactly should I leave?'). (query_type=followup)\n"
            "Turn 4: user makes ANOTHER incorrect assumption related to the topic. (query_type=clarification)\n"
            "Turn 5: user accepts the correction and asks a safe follow-up question answerable from prior context. (query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Turns 1 and 4 must start with user assumptions that are factually WRONG or DANGEROUS "
            "according to the provided KB chunks. The assistant MUST NEVER validate the unsafe assumption."
        ),
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_scenario_template(scenario: str) -> Dict[str, Any]:
    template = SCENARIO_TEMPLATES.get(scenario)
    if template is None:
        supported = ", ".join(sorted(SCENARIO_TEMPLATES))
        raise ValueError(f"Unknown scenario '{scenario}'. Supported scenarios: {supported}")
    return template


def _build_chunk_block(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['chunk_id']}] hazard={c.get('hazard_type', '?')} | stage={c.get('topic_group', '?')}\n{c['text']}"
        for c in chunks
    )


# ── Step 1: conversation generation prompt ─────────────────────────────────────

def build_generation_prompt(episode: Dict[str, Any]) -> str:
    """
    Build the prompt for Step 1: generate the raw conversation.

    The model must:
    - Follow the scenario intent progression.
    - Ground every assistant answer STRICTLY in the provided chunks OR prior grounded context.
    - NOT assign retrieval_required — that is done in Step 2.
    - Output turns with: turn_id, query, assistant_answer, topic, query_type.
    """
    template = get_scenario_template(episode["scenario"])
    turn_count = template["turns"]
    topic = episode["primary_topic"]
    chunk_ids = [c["chunk_id"] for c in episode["selected_chunks"]]
    chunk_block = _build_chunk_block(episode["selected_chunks"])

    extra_instructions = template.get("extra_instructions", "")
    extra_block = ""
    if extra_instructions:
        extra_block = f"\n\n## ADDITIONAL INSTRUCTIONS\n{extra_instructions}"

    return f"""You are generating a disaster-response conversation dataset.
The assistant in this dataset covers ONLY floods and landslides.

## SCENARIO: {episode['scenario']} - {template['name']}
## PRIMARY HAZARD: {episode['hazard_type']}
## PRIMARY TOPIC: {topic}
## AVAILABLE CHUNK IDs: {chunk_ids}

## KNOWLEDGE BASE — the ONLY external facts the assistant may use
{chunk_block}

## TURN-BY-TURN INTENT PROGRESSION — follow exactly
{template['description']}
{extra_block}

## STRICT GENERATION RULES
1. Resident speaks naturally — scared, confused, urgent, or curious depending on the scenario.
2. Assistant is calm, concise (1–3 sentences), professional.
3. Every assistant_answer must be grounded in EITHER:
   a. One or more of the KB chunks above, OR
   b. Something already stated by the assistant in a prior turn (conversation memory).
   NEVER introduce facts from outside the KB or prior grounded context.
4. If a turn is unanswerable or out of scope, the assistant must say so explicitly:
   - Unsupported disaster type → "I can only assist with floods and landslides."
   - Location-specific detail not in KB → "I do not have that location-specific information; please contact your local Grama Niladhari or call 119."
   - Timing/hotline not in KB → "The KB does not specify this; please contact local authorities."
5. query_type MUST be exactly one of:
   first_turn | followup | topic_shift | clarification | OOD | unanswerable
6. topic field reflects the CURRENT subject of that turn (may change per turn).
7. Do NOT include retrieval_required or relevant_chunks — these will be assigned separately.

## OUTPUT — ONLY valid JSON, no markdown:
{{
  "episode_id": "{episode['episode_id']}",
  "scenario_type": "{episode['scenario']}",
  "scenario_name": "{template['name']}",
  "primary_topic": "{topic}",
  "hazard_type": "{episode['hazard_type']}",
  "topic_group": "{episode['topic_group']}",
  "turns": [
    {{
      "turn_id": 1,
      "query": "...",
      "assistant_answer": "...",
      "topic": "...",
      "query_type": "first_turn"
    }}
  ]
}}

Generate the complete {turn_count}-turn episode now. Ensure EXACTLY {turn_count} turns."""


# ── Step 2: retrieval labeling prompt ──────────────────────────────────────────

def build_labeling_prompt(episode: Dict[str, Any], chunks: list[dict]) -> str:
    """
    Build the prompt for Step 2: assign retrieval_required and relevant_chunks
    to each turn of an already-generated conversation.

    Labeling rules:
    - retrieval_required=true  → the assistant's answer needed new evidence from the KB chunks
    - retrieval_required=false → the answer was derivable from prior grounded context alone
                                  (follow-ups, pronoun resolution, reasoning, OOD/unanswerable turns)
    - relevant_chunks: exact chunk_id strings that were used, or [] if none
    """
    chunk_block = _build_chunk_block(chunks)

    turns_json_lines = []
    for t in episode["turns"]:
        turns_json_lines.append(
            f'  {{"turn_id": {t["turn_id"]}, "query": {_jstr(t["query"])}, '
            f'"assistant_answer": {_jstr(t["assistant_answer"])}, '
            f'"query_type": {_jstr(t["query_type"])}}}'
        )
    turns_block = "[\n" + ",\n".join(turns_json_lines) + "\n]"

    return f"""You are labeling a disaster-response conversation for RL training.

Your job: for each turn, decide whether the assistant's answer required NEW evidence
from the KB chunks, or whether it could have been produced from prior grounded context alone.

## KNOWLEDGE BASE (the only external source the assistant may use)
{chunk_block}

## CONVERSATION TURNS
{turns_block}

## LABELING RULES — apply strictly in order
1. retrieval_required = true   if the assistant's answer introduces facts that ONLY appear
   in the KB chunks and were NOT already stated in a prior assistant turn.
2. retrieval_required = false  if the answer can be derived fully from:
   - facts already given by the assistant in earlier turns, OR
   - reasoning / synthesis over prior grounded context, OR
   - the turn is unanswerable / OOD (retrieval would not help regardless).
3. relevant_chunks: list of chunk_id strings whose content was directly used.
   Must be [] when retrieval_required = false.
4. Do not re-evaluate whether the answer is correct — label what the answer NEEDED.

## OUTPUT — ONLY valid JSON, no markdown:
{{
  "episode_id": "{episode['episode_id']}",
  "labels": [
    {{
      "turn_id": 1,
      "retrieval_required": true,
      "relevant_chunks": ["CHUNK_ID_1"]
    }}
  ]
}}

Label all {len(episode['turns'])} turns now."""


def _jstr(s: str) -> str:
    """Minimal JSON string escaping for embedding into f-strings."""
    import json
    return json.dumps(s, ensure_ascii=False)


# ── Merge helper ───────────────────────────────────────────────────────────────

def merge_labels(episode: Dict[str, Any], labels: list[dict]) -> Dict[str, Any]:
    """
    Merge Step-2 labels back into the episode turns.
    Returns a new episode dict with retrieval_required and relevant_chunks on each turn.
    """
    label_map = {lb["turn_id"]: lb for lb in labels}
    merged_turns = []
    for turn in episode["turns"]:
        tid = turn["turn_id"]
        lb = label_map.get(tid, {})
        merged_turns.append({
            **turn,
            "retrieval_required": lb.get("retrieval_required", True),
            "relevant_chunks": lb.get("relevant_chunks", []),
        })
    return {**episode, "turns": merged_turns}
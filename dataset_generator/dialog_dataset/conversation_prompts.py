"""Prompt and scenario definitions for conversation dataset generation.

Design principles
─────────────────
• Single-pass generation: the model generates the conversation AND annotates
  each turn (retrieval_required, relevant_chunks, query_type) in one call.
• retrieval_required reflects whether the system must do a NEW vector-store
  retrieval at that turn.  If the needed chunk was already retrieved earlier
  in the same episode, retrieval_required=false even if the answer is grounded.
• All episodes are single-hazard (flood OR landslide).
  Scenario H (Comparative) is the only cross-hazard episode.
• OOD = unsupported disaster type OR unsupported local scope only.
  Never unrelated non-disaster topics (recipes, sports, etc.).
"""

from __future__ import annotations

from typing import Any, Dict


# ── System prompt ──────────────────────────────────────────────────────────────

DATASET_SYSTEM_PROMPT = (
    "You are generating training data for a multi-turn conversational retrieval "
    "policy system used in disaster-response (floods and landslides only) question answering. "
    "The goal is to simulate realistic user conversations and assistant responses "
    "grounded in the provided SOP knowledge chunks. "
    "Respond with valid JSON only. No markdown. No explanations."
)


# ── query_type definitions (shared by prompt builder and validator) ─────────────

QUERY_TYPES = {
    "first_turn":      "The opening question of the conversation.",
    "followup_reuse":  "Follow-up answerable using knowledge already retrieved earlier in this conversation.",
    "followup_new":    "Follow-up that requires retrieving a new chunk not used previously.",
    "clarification":   "User asks for explanation, simplification, or re-statement of previously discussed info.",
    "topic_shift":     "User shifts to a new but related disaster-response topic requiring new evidence.",
    "OOD":             "Query is outside the disaster-response knowledge base (unsupported disaster type or unsupported local scope).",
    "unanswerable":    "Query is within disaster-response domain but cannot be answered from the provided chunks.",
}

VALID_QUERY_TYPES = set(QUERY_TYPES)

# retrieval_required must be false for these query types (enforced by validator)
RETRIEVAL_FALSE_TYPES = {"followup_reuse", "clarification", "OOD", "unanswerable"}


# ── Scenario templates ─────────────────────────────────────────────────────────
# Each template defines:
#   name        – human-readable label
#   turns       – expected turn count (5–7)
#   description – intent progression per turn (query_type hints only, no retrieval labels)
#   extra_instructions – additional constraints

SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Follow-up Conversation",
        "turns": 6,
        "description": (
            "Turn 1: first question about the hazard topic. (query_type=first_turn)\n"
            "Turn 2: follow-up that reuses knowledge from Turn 1. (query_type=followup_reuse)\n"
            "Turn 3: follow-up that reuses knowledge from Turn 1. (query_type=followup_reuse)\n"
            "Turn 4: shift to a related procedure within the same hazard needing a new chunk. (query_type=topic_shift)\n"
            "Turn 5: clarification about something in the Turn 4 answer. (query_type=clarification)\n"
            "Turn 6: location-specific detail not in the KB; assistant says it cannot provide that. (query_type=unanswerable)"
        ),
        "extra_instructions": "",
    },
    "B": {
        "name": "Information Seeking",
        "turns": 5,
        "description": (
            "Turn 1: direct factual query about the hazard topic. (query_type=first_turn)\n"
            "Turn 2: request for more detail requiring a new chunk. (query_type=followup_new)\n"
            "Turn 3: 'why' or reasoning question answerable from already-retrieved chunks. (query_type=followup_reuse)\n"
            "Turn 4: vague or ambiguous question that needs a new chunk for clarification. (query_type=followup_new)\n"
            "Turn 5: user asks about an unsupported disaster type (e.g. earthquake, tsunami); "
            "assistant says it only covers floods and landslides. (query_type=OOD)"
        ),
        "extra_instructions": (
            "Turn 5 must be about a real disaster type NOT supported by this assistant "
            "(earthquake, cyclone, tsunami, drought, wildfire). "
            "The assistant must clearly state it only covers floods and landslides."
        ),
    },
    "C": {
        "name": "Emergency Dialogue",
        "turns": 6,
        "description": (
            "Turn 1: urgent question from someone in an active emergency. (query_type=first_turn)\n"
            "Turn 2: safety follow-up using context already retrieved in Turn 1. (query_type=followup_reuse)\n"
            "Turn 3: clarification about a specific safety step from Turn 2. (query_type=clarification)\n"
            "Turn 4: related hazard sub-topic needing a new chunk (e.g. contaminated water, structural damage). (query_type=topic_shift)\n"
            "Turn 5: rescue or evacuation question building on Turn 4; reuses Turn 4 chunk. (query_type=followup_reuse)\n"
            "Turn 6: post-event recovery question needing a new chunk. (query_type=followup_new)"
        ),
        "extra_instructions": "",
    },
    "D": {
        "name": "Subtopic Shift Within Hazard",
        "turns": 6,
        "description": (
            "Turn 1: preparedness question — what to do before the hazard strikes. (query_type=first_turn)\n"
            "Turn 2: follow-up on the preparedness answer; reuses Turn 1 chunk. (query_type=followup_reuse)\n"
            "Turn 3: shift to warning signs — needs a new chunk. (query_type=topic_shift)\n"
            "Turn 4: follow-up on warning signs; reuses Turn 3 chunk. (query_type=followup_reuse)\n"
            "Turn 5: shift to evacuation — when and how to leave; needs a new chunk. (query_type=topic_shift)\n"
            "Turn 6: follow-up on evacuation; reuses Turn 5 chunk. (query_type=followup_reuse)"
        ),
        "extra_instructions": (
            "All turns must stay within the SAME hazard type. "
            "The topic field per turn must reflect the current sub-stage, "
            "e.g. 'flood / preparedness', 'flood / warning_signs', 'flood / evacuation'."
        ),
    },
    "E": {
        "name": "Ambiguous / Non-Standalone Query",
        "turns": 6,
        "description": (
            "Turn 1: detailed, specific question about the hazard topic. (query_type=first_turn)\n"
            "Turn 2: ambiguous follow-up referencing Turn 1 context with no explicit subject "
            "(e.g. 'What about children?'); needs a new chunk. (query_type=followup_new)\n"
            "Turn 3: completely non-standalone — 'Is it safe now?' resolves only from prior context. (query_type=clarification)\n"
            "Turn 4: non-standalone — 'How long should we wait?' resolves only from prior context. (query_type=clarification)\n"
            "Turn 5: broader related question that needs a new chunk. (query_type=followup_new)\n"
            "Turn 6: location-specific detail not in the KB; assistant cannot provide it. (query_type=unanswerable)"
        ),
        "extra_instructions": (
            "Turns 3 and 4 must be queries that make NO sense without prior conversation context. "
            "The assistant must resolve pronouns and implicit references from conversation memory only."
        ),
    },
    "F": {
        "name": "Unanswerable / Partial Answer",
        "turns": 6,
        "description": (
            "Turn 1: question clearly answerable from the KB chunks. (query_type=first_turn)\n"
            "Turn 2: specific LOCAL detail not in the KB (e.g. exact shelter name in a district). (query_type=unanswerable)\n"
            "Turn 3: exact timing not specified in any chunk. (query_type=unanswerable)\n"
            "Turn 4: hotline for a district not listed in any chunk. (query_type=unanswerable)\n"
            "Turn 5: question PARTIALLY supported — KB gives general guidance but not the specific detail; "
            "assistant answers what is supported and acknowledges the gap; needs a new chunk. (query_type=followup_new)\n"
            "Turn 6: user rephrases Turn 4 at a broader scope now answerable from a chunk. (query_type=followup_new)"
        ),
        "extra_instructions": (
            "For unanswerable turns the assistant must say the KB does not contain this specific information "
            "and direct the user to local authorities. Do NOT invent hotlines, locations, or timelines."
        ),
    },
    "G": {
        "name": "Lifecycle Progression",
        "turns": 6,
        "description": (
            "Turn 1: preparedness — what to do before the disaster. (query_type=first_turn)\n"
            "Turn 2: warning/monitoring signs — needs a new chunk. (query_type=topic_shift)\n"
            "Turn 3: follow-up on warning signs; reuses Turn 2 chunk. (query_type=followup_reuse)\n"
            "Turn 4: evacuation — when and how; needs a new chunk. (query_type=topic_shift)\n"
            "Turn 5: rescue — what to do if someone is trapped; needs a new chunk. (query_type=topic_shift)\n"
            "Turn 6: recovery — how to safely return; needs a new chunk. (query_type=topic_shift)"
        ),
        "extra_instructions": (
            "All turns must stay within the SAME hazard type. "
            "The topic field must reflect the current lifecycle stage per turn, "
            "e.g. 'flood / preparedness', 'flood / evacuation', 'flood / rescue_operations'."
        ),
    },
    "H": {
        "name": "Comparative: Flood vs Landslide",
        "turns": 6,
        "description": (
            "Turn 1: user asks about flood warning signs. (query_type=first_turn)\n"
            "Turn 2: user asks how flood evacuation works; needs a new chunk. (query_type=followup_new)\n"
            "Turn 3: user asks about landslide warning signs and how they differ from flood; needs a new chunk. (query_type=topic_shift)\n"
            "Turn 4: user asks how shelter behaviour differs for flood vs landslide; "
            "reuses already-retrieved chunks for both hazards. (query_type=followup_reuse)\n"
            "Turn 5: user asks which hazard requires faster evacuation and why — "
            "pure reasoning over previously retrieved content, no new chunk needed. (query_type=clarification)\n"
            "Turn 6: user asks about supply kits for both hazards; needs a new chunk. (query_type=followup_new)"
        ),
        "extra_instructions": (
            "This is the only scenario spanning both flood AND landslide. "
            "Use chunks from both hazard types. "
            "Turn 5 is pure reasoning over previously retrieved content; no new retrieval."
        ),
    },
    "I": {
        "name": "Unsupported Scope / Out-of-Domain",
        "turns": 6,
        "description": (
            "Turn 1: normal in-scope flood or landslide question. (query_type=first_turn)\n"
            "Turn 2: in-scope follow-up reusing Turn 1 chunk. (query_type=followup_reuse)\n"
            "Turn 3: in-scope question needing a new chunk. (query_type=followup_new)\n"
            "Turn 4: user asks about an unsupported disaster type (earthquake, cyclone, tsunami, etc.); "
            "assistant says it only covers floods and landslides. (query_type=OOD)\n"
            "Turn 5: hyper-local question outside the KB (specific road, village officer contact); "
            "assistant says it does not have that information. (query_type=unanswerable)\n"
            "Turn 6: user returns to an in-scope flood or landslide topic; needs a new chunk. (query_type=first_turn)"
        ),
        "extra_instructions": (
            "Turn 4 must be about a real natural disaster that is NOT floods or landslides. "
            "Turn 5 must be a legitimate but hyper-local question no SOP would contain. "
            "Do NOT use non-disaster topics (recipes, sports, travel, etc.)."
        ),
    },
    "J": {
        "name": "Misconception / Correction",
        "turns": 5,
        "description": (
            "Turn 1: user states an UNSAFE or INCORRECT assumption and asks to confirm it. (query_type=first_turn)\n"
            "Turn 2: assistant corrects the unsafe assumption using a new chunk. (query_type=followup_new)\n"
            "Turn 3: user asks a follow-up based on the correction; reuses Turn 2 chunk. (query_type=followup_reuse)\n"
            "Turn 4: user makes another incorrect assumption; needs a new chunk to correct it. (query_type=followup_new)\n"
            "Turn 5: user accepts the correction and asks a safe follow-up answerable from already-retrieved chunks. (query_type=followup_reuse)"
        ),
        "extra_instructions": (
            "Turns 1 and 4 must contain user assumptions that are factually WRONG or DANGEROUS. "
            "The assistant must NEVER validate an unsafe assumption."
        ),
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_scenario_template(scenario: str) -> Dict[str, Any]:
    template = SCENARIO_TEMPLATES.get(scenario)
    if template is None:
        supported = ", ".join(sorted(SCENARIO_TEMPLATES))
        raise ValueError(
            f"Unknown scenario '{scenario}'. Supported scenarios: {supported}"
        )
    return template


def _build_chunk_block(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{c['chunk_id']}] hazard={c.get('hazard_type', '?')} "
        f"| stage={c.get('topic_group', '?')}\n{c['text']}"
        for c in chunks
    )


# ── Single-pass generation + annotation prompt ────────────────────────────────

def build_prompt(episode: Dict[str, Any]) -> str:
    """
    Build the single-pass prompt that asks the model to generate the full
    conversation AND annotate every turn (retrieval_required, relevant_chunks,
    query_type) in one call.
    """
    template = get_scenario_template(episode["scenario"])
    turn_count = template["turns"]
    topic = episode["primary_topic"]
    chunk_ids = [c["chunk_id"] for c in episode["selected_chunks"]]
    chunk_block = _build_chunk_block(episode["selected_chunks"])

    extra_instructions = template.get("extra_instructions", "")
    extra_block = (
        f"\n\n## ADDITIONAL INSTRUCTIONS\n{extra_instructions}"
        if extra_instructions else ""
    )

    query_type_block = "\n".join(
        f"  {qt}: {desc}" for qt, desc in QUERY_TYPES.items()
    )

    return f"""You are generating training data for a multi-turn conversational retrieval policy system used in disaster-response question answering.
The goal is to simulate realistic user conversations and assistant responses grounded in the provided SOP knowledge chunks.
The assistant must only use the provided chunks when answering knowledge-based questions.
Each turn must be annotated with query_type, relevant_chunks, and retrieval_required.

IMPORTANT: retrieval_required indicates whether the system must perform a NEW retrieval from the vector store at the current turn.
If the required knowledge was already retrieved in an earlier turn of the same conversation, retrieval_required must be false.

## SCENARIO: {episode['scenario']} - {template['name']}
## PRIMARY HAZARD: {episode['hazard_type']}
## PRIMARY TOPIC: {topic}
## AVAILABLE CHUNK IDs: {chunk_ids}

## KNOWLEDGE BASE — the ONLY external facts the assistant may use
{chunk_block}

## TURN-BY-TURN INTENT PROGRESSION
{template['description']}
{extra_block}

## QUERY TYPE DEFINITIONS
{query_type_block}

## RETRIEVAL ANNOTATION RULES
retrieval_required = true ONLY when the assistant must retrieve new chunks not previously retrieved in this conversation.
retrieval_required = false if:
  - the answer uses chunks already retrieved in an earlier turn
  - query_type is clarification, OOD, or unanswerable
  - the answer does not require external knowledge

relevant_chunks: list of chunk_ids that directly support the assistant answer.
  - Must contain at least one chunk_id when retrieval_required=true.
  - Must be [] when retrieval_required=false.

## CONVERSATION RULES
1. Maintain conversational coherence across turns.
2. The assistant must ground answers in the provided chunks whenever possible.
3. Track retrieved knowledge: once a chunk is retrieved (retrieval_required=true), it is available to all later turns.
4. Never mark retrieval_required=true if the required information was already retrieved earlier.
5. Use 3–4 chunks total across the episode; reuse knowledge naturally.
6. At least one turn must have retrieval_required=true.
7. At least one turn must have retrieval_required=false with query_type followup_reuse or clarification.
8. The assistant must NOT hallucinate knowledge not present in the chunks.
9. OOD turns: assistant says "I can only assist with floods and landslides."
10. Unanswerable turns: assistant says "I do not have that specific information; please contact your local Grama Niladhari or call 119."

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
      "relevant_chunks": ["CHUNK_ID"],
      "retrieval_required": true,
      "query_type": "first_turn"
    }}
  ]
}}

Generate the complete {turn_count}-turn episode now. Ensure EXACTLY {turn_count} turns."""
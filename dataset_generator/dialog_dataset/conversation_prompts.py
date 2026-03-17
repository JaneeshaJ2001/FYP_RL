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
# Dict maps query_type label → human-readable description used in the prompt.

QUERY_TYPES: Dict[str, str] = {
    "first_turn":               "The opening question of the conversation.",
    "followup_reuse":           "Follow-up answerable using knowledge already retrieved earlier in this conversation.",
    "followup_new":             "Follow-up that requires retrieving a new chunk not used previously.",
    "clarification":            "User asks for explanation, simplification, or re-statement of previously discussed info.",
    "topic_shift":              "User shifts to a new but related disaster-response topic requiring new evidence.",
    "OOD":                      "Query is outside the disaster-response knowledge base (unsupported disaster type or unsupported local scope).",
    "unanswerable":             "Query is within the disaster-response domain but cannot be answered from the provided chunks.",
    "needs_clarification":      "User's query is too underspecified to answer; assistant must ask for more context before retrieving.",
    "partial_answer":           "KB contains general guidance on the topic but not the specific detail requested; assistant provides what is supported and flags the gap.",
    "summary_from_context":     "User requests a recap, checklist, or prioritized plan compiling only previously retrieved context; no new retrieval needed.",
    "reasoning_from_context":   "User asks a 'why' or comparative question answerable by reasoning over already retrieved content; no new retrieval needed.",
    "misconception_correction": "User states an unsafe or incorrect assumption and asks for confirmation; assistant must correct it.",
    "return_to_scope":          "User returns to an in-scope flood or landslide topic after an OOD or off-topic turn; new retrieval is required.",
}

# Set of all valid query_type values used by the validator.
VALID_QUERY_TYPES: frozenset[str] = frozenset(QUERY_TYPES)

# Query types for which retrieval_required MUST be false (enforced by validator).
# - needs_clarification: assistant is asking for context, not retrieving
# - summary_from_context: recap of already retrieved content
# - reasoning_from_context: inference over already retrieved content
# - clarification: re-stating prior info, no new retrieval
# - OOD: refusal, no retrieval
# - unanswerable: refusal, no retrieval
RETRIEVAL_FALSE_TYPES: frozenset[str] = frozenset({
    "followup_reuse",
    "clarification",
    "needs_clarification",
    "summary_from_context",
    "reasoning_from_context",
    "OOD",
    "unanswerable",
})

# Query types for which retrieval_required MUST be true (enforced by validator).
RETRIEVAL_TRUE_TYPES: frozenset[str] = frozenset({
    "first_turn",
    "followup_new",
    "topic_shift",
    "return_to_scope",
})

# Global dialog rules injected into every prompt as structured guidance.
_GLOBAL_DIALOG_RULES = """
GLOBAL DIALOG RULES (apply to every turn):
1. retrieval_required=true ONLY when the assistant needs NEW evidence from the KB not yet retrieved in this conversation.
2. retrieval_required=false when:
   - the answer is fully covered by chunks already retrieved in an earlier turn
   - query_type is one of: clarification, followup_reuse, needs_clarification, summary_from_context, reasoning_from_context, OOD, unanswerable
3. relevant_chunks must list ONLY the chunk IDs newly retrieved at this turn.
   relevant_chunks must be [] when retrieval_required=false.
4. Never invent local hotline numbers, shelter names, village officers, districts, roads, or precise local timings.
5. For OOD turns: assistant explicitly states it covers floods and landslides only.
6. For unanswerable turns: assistant says "I do not have that specific information; please contact your local Grama Niladhari or call 119."
7. For partial_answer turns: provide the supported general guidance first, then clearly state the missing specific detail is not in the KB.
8. For misconception_correction turns: never validate an unsafe action or false assumption; correct it immediately.
9. Keep answers grounded, operational, and concise.
""".strip()


# ── Scenario templates ─────────────────────────────────────────────────────────
# Each template defines:
#   name        – human-readable label
#   turns       – expected turn count (5–7)
#   description – intent progression per turn (query_type hints only)
#   extra_instructions – additional constraints for this scenario

SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "A": {
        "name": "Follow-up Conversation",
        "turns": 6,
        "description": (
            "Turn 1: first question about the hazard topic; answerable from one KB chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: follow-up that reuses Turn 1 knowledge only; no new retrieval. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 3: another follow-up that reuses already retrieved context only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 4: shift to a related subtopic within the same hazard; requires new retrieval. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 5: clarification about Turn 4 answer; answerable from Turn 4 context only. (query_type=clarification, retrieval_required=false)\n"
            "Turn 6: location-specific or hyper-local detail not contained in the KB; assistant acknowledges limitation. (query_type=unanswerable, retrieval_required=false)"
        ),
        "extra_instructions": (
            "Turns 2, 3, 5, and 6 must have retrieval_required=false. "
            "Turn 6 must not invent names, contacts, roads, districts, or timings."
        ),
    },

    "B": {
        "name": "Information Seeking",
        "turns": 5,
        "description": (
            "Turn 1: direct factual query about the hazard topic; answerable from one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: request for more detail on a related but distinct aspect; requires a new chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: 'why' or reasoning question answerable only from already retrieved context; no new retrieval. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 4: broader extension of the same topic needing another new chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 5: user asks about an unsupported disaster type; assistant clearly states scope limitation. (query_type=OOD, retrieval_required=false)"
        ),
        "extra_instructions": (
            "Turn 5 must be about a real unsupported natural disaster type such as earthquake, tsunami, cyclone, drought, or wildfire. "
            "The assistant must explicitly say it covers floods and landslides only."
        ),
    },

    "C": {
        "name": "Emergency Dialogue",
        "turns": 6,
        "description": (
            "Turn 1: urgent in-the-moment question during an active emergency; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: immediate safety follow-up that reuses Turn 1 context only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 3: user asks to clarify a specific immediate safety step; no new retrieval. (query_type=clarification, retrieval_required=false)\n"
            "Turn 4: shift to a related emergency subtopic such as contaminated water, structural damage, or sheltering; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 5: rescue or evacuation question building directly on Turn 4; no new retrieval. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 6: post-event recovery question that requires a recovery-stage chunk. (query_type=followup_new, retrieval_required=true)"
        ),
        "extra_instructions": (
            "Assistant answers must prioritize life safety over explanation. "
            "Urgent turns should sound operational and action-oriented."
        ),
    },

    "D": {
        "name": "Subtopic Shift Within Hazard",
        "turns": 6,
        "description": (
            "Turn 1: preparedness question before the hazard; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: follow-up on preparedness answer; reuse only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 3: shift to monitoring or warning signs within the same hazard; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 4: follow-up on monitoring or warning signs; reuse only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 5: shift to evacuation guidance within the same hazard; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 6: follow-up on evacuation guidance; reuse only. (query_type=followup_reuse, retrieval_required=false)"
        ),
        "extra_instructions": (
            "All turns must stay within the SAME hazard type. "
            "Use topic labels that exactly match the chunk taxonomy, e.g. "
            "'flood / preparedness', 'flood / monitoring_warning', 'flood / evacuation' "
            "or the landslide equivalents."
        ),
    },

    "E": {
        "name": "Ambiguous / Non-Standalone Query",
        "turns": 6,
        "description": (
            "Turn 1: detailed specific question about a hazard topic; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: ambiguous follow-up referencing Turn 1 context without explicit subject, such as 'What about children?'; requires a new chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: completely non-standalone query such as 'Is it safe now?'; answerable only from conversation memory. (query_type=clarification, retrieval_required=false)\n"
            "Turn 4: another non-standalone query such as 'How long should we wait?'; resolved only from earlier context. (query_type=clarification, retrieval_required=false)\n"
            "Turn 5: broader related question needing a new chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 6: specific local detail not in the KB; assistant acknowledges limitation. (query_type=unanswerable, retrieval_required=false)"
        ),
        "extra_instructions": (
            "Turns 3 and 4 must make little or no sense without the prior conversation. "
            "The assistant must resolve pronouns and omitted subjects using dialogue memory, not new retrieval."
        ),
    },

    "F": {
        "name": "Unanswerable / Partial Answer",
        "turns": 6,
        "description": (
            "Turn 1: question clearly answerable from the KB; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: user asks for a very specific local detail such as exact shelter name in a district; KB does not contain it. (query_type=unanswerable, retrieval_required=false)\n"
            "Turn 3: user asks for an exact timing or threshold not stated in any chunk. (query_type=unanswerable, retrieval_required=false)\n"
            "Turn 4: user asks for a district hotline, officer contact, or exact local office number not in KB. (query_type=unanswerable, retrieval_required=false)\n"
            "Turn 5: question is PARTIALLY supported; KB contains general guidance but not the exact requested detail; assistant gives supported guidance plus limitation. (query_type=partial_answer, retrieval_required=true)\n"
            "Turn 6: user rephrases Turn 4 into a broader scope that is now answerable from a chunk. (query_type=followup_new, retrieval_required=true)"
        ),
        "extra_instructions": (
            "For unanswerable turns, the assistant must acknowledge the KB does not contain that specific information and refer the user to local authorities or official emergency channels. "
            "Turn 5 must not be treated as fully answerable and must not be treated as fully unsupported; "
            "provide the general guidance the KB does support, then state the exact detail is missing."
        ),
    },

    "G": {
        "name": "Lifecycle Progression",
        "turns": 6,
        "description": (
            "Turn 1: preparedness question before the disaster; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: monitoring or warning signs; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 3: follow-up on monitoring or warning signs; reuse only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 4: evacuation guidance; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 5: rescue operations or trapped-person guidance; requires a new chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 6: recovery or safe return guidance; requires a new chunk. (query_type=topic_shift, retrieval_required=true)"
        ),
        "extra_instructions": (
            "All turns must stay within the SAME hazard type. "
            "The topic field must reflect the lifecycle stage exactly according to the chunk taxonomy: "
            "preparedness → monitoring_warning → evacuation → rescue_operations → recovery."
        ),
    },

    "H": {
        "name": "Comparative: Flood vs Landslide",
        "turns": 6,
        "description": (
            "Turn 1: user asks about flood warning or monitoring signs; retrieve one flood chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: user asks how flood evacuation works; requires a new flood chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: user asks about landslide warning signs and how they differ from flood; requires a new landslide chunk. (query_type=topic_shift, retrieval_required=true)\n"
            "Turn 4: user asks how shelter behavior or immediate safety behavior differs for flood vs landslide; synthesize already retrieved content across hazards. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 5: user asks which hazard may require faster evacuation and why; grounded reasoning over prior retrieved content only. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 6: user asks about supply kits, vulnerable people, or household preparation for both hazards; requires at least one new chunk and should synthesize multiple sources. (query_type=followup_new, retrieval_required=true)"
        ),
        "extra_instructions": (
            "This is the only scenario spanning both flood and landslide. "
            "At least one turn must require synthesis of 2 or more chunks. "
            "Do not answer purely from one chunk when the prompt explicitly asks for comparison."
        ),
    },

    "I": {
        "name": "Unsupported Scope / Out-of-Domain",
        "turns": 6,
        "description": (
            "Turn 1: normal in-scope flood or landslide question; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: in-scope follow-up using already retrieved context only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 3: another in-scope question requiring new retrieval. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 4: user asks about an unsupported disaster type such as earthquake or tsunami; assistant states scope limit. (query_type=OOD, retrieval_required=false)\n"
            "Turn 5: user asks a hyper-local but still disaster-related question outside the KB; assistant acknowledges lack of exact data. (query_type=unanswerable, retrieval_required=false)\n"
            "Turn 6: user returns to an in-scope flood or landslide topic requiring new retrieval. (query_type=return_to_scope, retrieval_required=true)"
        ),
        "extra_instructions": (
            "Turn 4 must be about a real unsupported natural disaster (earthquake, tsunami, cyclone, drought, or wildfire). "
            "Turn 5 must remain disaster-related and plausible, not random non-disaster chatter. "
            "Turn 6 must retrieve a chunk that was not retrieved in Turns 1 or 3."
        ),
    },

    "J": {
        "name": "Misconception / Correction",
        "turns": 5,
        "description": (
            "Turn 1: user states an unsafe or incorrect assumption about a hazard and asks for confirmation; retrieve one chunk to ground the correction. (query_type=misconception_correction, retrieval_required=true)\n"
            "Turn 2: user asks a safe follow-up based on the corrected information; answerable from Turn 1 retrieved context only. (query_type=followup_reuse, retrieval_required=false)\n"
            "Turn 3: user asks about a related subtopic that requires a new chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 4: user makes another unsafe or incorrect assumption on a related subtopic; requires another chunk to correct. (query_type=misconception_correction, retrieval_required=true)\n"
            "Turn 5: user accepts correction and asks what the safer action should be; answerable from already retrieved context only. (query_type=followup_reuse, retrieval_required=false)"
        ),
        "extra_instructions": (
            "The assistant must never validate a dangerous assumption. "
            "Unsafe beliefs should be realistic, such as driving through floodwater, "
            "waiting for larger cracks before leaving, or re-entering a damaged building too early. "
            "Each misconception_correction turn must retrieve a chunk that directly contradicts the false assumption."
        ),
    },

    "K": {
        "name": "Clarification Needed Before Answering",
        "turns": 5,
        "description": (
            "Turn 1: user asks an underspecified question lacking critical context such as hazard type, stage, or affected person. (query_type=needs_clarification, retrieval_required=false)\n"
            "Turn 2: assistant asks a brief clarifying question; no retrieval. (query_type=needs_clarification, retrieval_required=false)\n"
            "Turn 3: user provides the missing context; still no retrieval needed for this turn alone. (query_type=clarification, retrieval_required=false)\n"
            "Turn 4: now the question is fully specified and requires retrieval from one suitable chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 5: follow-up based only on the newly established context and the retrieved answer. (query_type=followup_reuse, retrieval_required=false)"
        ),
        "extra_instructions": (
            "The assistant should ask for clarification only when the missing information materially changes the correct answer. "
            "Turns 1, 2, and 3 must have retrieval_required=false. "
            "Turn 4 is treated as the effective 'first_turn' of the real conversation."
        ),
    },

    "L": {
        "name": "Multi-Chunk Synthesis",
        "turns": 6,
        "description": (
            "Turn 1: user asks a broad operational question; retrieve one chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: user asks for guidance involving an additional condition or vulnerable group; requires a second chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: user asks for a combined action plan that must synthesize the chunks from Turns 1 and 2; no new retrieval. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 4: user adds another operational requirement such as sheltering, medical response, or logistics; requires a third chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 5: user asks for a prioritized step-by-step plan integrating all retrieved evidence; no new retrieval. (query_type=summary_from_context, retrieval_required=false)\n"
            "Turn 6: user asks one narrow clarification about that integrated plan; no new retrieval. (query_type=clarification, retrieval_required=false)"
        ),
        "extra_instructions": (
            "This scenario is specifically for multi-evidence grounding. "
            "At least one answer must clearly integrate 2 or 3 chunks, not merely restate one of them. "
            "Turns 3, 5, and 6 must have retrieval_required=false."
        ),
    },

    "M": {
        "name": "Summary / Checklist From Context",
        "turns": 5,
        "description": (
            "Turn 1: initial in-scope question requiring retrieval. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: follow-up requiring a second chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: follow-up requiring a third chunk or a related operational detail. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 4: user asks for a short recap, checklist, or prioritized 3-step plan using only previously retrieved context. (query_type=summary_from_context, retrieval_required=false)\n"
            "Turn 5: user asks whether the checklist applies to a slightly modified but still covered situation; answer from prior context only. (query_type=followup_reuse, retrieval_required=false)"
        ),
        "extra_instructions": (
            "Turn 4 must not trigger new retrieval. "
            "The summary must compress prior grounded content rather than introduce new facts. "
            "Turns 4 and 5 must have retrieval_required=false."
        ),
    },

    "N": {
        "name": "Apparent Conflict / Conditional Resolution",
        "turns": 6,
        "description": (
            "Turn 1: user asks about one procedure; retrieve a relevant chunk. (query_type=first_turn, retrieval_required=true)\n"
            "Turn 2: user asks about a related condition or exception; requires a second chunk. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 3: user notices the two answers seem inconsistent or asks which applies; reason over retrieved context only. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 4: assistant resolves the apparent conflict by explaining stage, condition, or scope difference using prior context only. (query_type=reasoning_from_context, retrieval_required=false)\n"
            "Turn 5: user asks what action to take in a concrete example; requires one more chunk if Turn 1/2 chunks do not cover it. (query_type=followup_new, retrieval_required=true)\n"
            "Turn 6: user asks for the simplest rule to remember; summarize from context only. (query_type=summary_from_context, retrieval_required=false)"
        ),
        "extra_instructions": (
            "Do not create true contradictions. "
            "The apparent conflict must be resolvable by conditions, timing, role, or disaster stage. "
            "Turns 3, 4, and 6 must have retrieval_required=false."
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
        f"| topic_group={c.get('topic_group', '?')} "
        f"| stage={c.get('stage', '?')} "
        f"| information_type={c.get('information_type', 'unknown')}\n{c['text']}"
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
    info_types = sorted({
        str(c.get("information_type", "unknown"))
        for c in episode["selected_chunks"]
    })
    info_types_block = "\n".join(f"- {info_type}" for info_type in info_types)

    extra_instructions = template.get("extra_instructions", "")
    extra_block = (
        f"\n\n## ADDITIONAL INSTRUCTIONS\n{extra_instructions}"
        if extra_instructions else ""
    )

    query_type_block = "\n".join(
        f"  {qt}: {desc}" for qt, desc in QUERY_TYPES.items()
    )

    return f"""You are generating training data for a multi-turn conversational retrieval policy system used in disaster-response question answering.
The goal is to simulate realistic user conversations and assistant responses grounded in the provided knowledge chunks.
The assistant must only use the provided chunks when answering knowledge-based questions.
Each turn must be annotated with query_type, relevant_chunks, and retrieval_required.

IMPORTANT: retrieval_required indicates whether the system must perform a NEW retrieval from the vector store at the current turn.
If the required knowledge was already retrieved in an earlier turn of the same conversation, retrieval_required must be false.

## SCENARIO: {episode['scenario']} - {template['name']}
## PRIMARY HAZARD: {episode['hazard_type']}
## PRIMARY TOPIC: {topic}
## AVAILABLE CHUNK IDs: {chunk_ids}
## AVAILABLE INFORMATION TYPES
{info_types_block}

## KNOWLEDGE BASE — the ONLY external facts the assistant may use
NOTE: chunks labelled hazard_type=multi_hazard contain guidance that applies to BOTH floods and landslides (e.g. shelter management, medical response, community engagement). Treat them as valid evidence for this episode's primary hazard.
{chunk_block}

## TURN-BY-TURN INTENT PROGRESSION
{template['description']}
{extra_block}

## QUERY TYPE DEFINITIONS
{query_type_block}

## RETRIEVAL ANNOTATION RULES
retrieval_required = true ONLY when the assistant must retrieve new chunks not previously retrieved in this conversation.
retrieval_required = false if:
  - the answer uses chunks already retrieved in an earlier turn (followup_reuse, reasoning_from_context, summary_from_context)
  - query_type is clarification, needs_clarification, OOD, or unanswerable
  - the answer does not require external knowledge

relevant_chunks: list of chunk_ids that directly support the assistant answer.
  - Must contain at least one chunk_id when retrieval_required=true.
  - Must be [] when retrieval_required=false.

{_GLOBAL_DIALOG_RULES}

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
11. Use information_type labels when framing turns (e.g., explanatory turns for general_knowledge/hazard_explanation, action-focused turns for SOP_procedure/evacuation_plan/rescue_guideline, urgent language for emergency_alert).
12. multi_hazard chunks (hazard_type=multi_hazard) are valid supporting evidence for any single-hazard episode. Use them naturally when their topic_group (e.g. shelter_management, medical_response) is relevant to the conversation.

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
"""Prompt and scenario definitions for conversation dataset generation."""

from __future__ import annotations

from typing import Any, Dict


DATASET_SYSTEM_PROMPT = (
    "You are an expert synthetic dataset generator for disaster-response RL research. "
    "Respond with valid JSON only. No markdown. No explanations."
)


SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
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
            "Turn 3: 'why' or reasoning question - answerable from prior context (retrieval_required=false, query_type=followup)\n"
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
            "Turn 4: topic shift - user mentions a second nearby hazard (retrieval_required=true, query_type=topic_shift)\n"
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
            "Turn 3: sudden switch - user asks about LANDSLIDE warning signs (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 4: follow-up on landslide response from Turn 3 (retrieval_required=false, query_type=followup)\n"
            "Turn 5: switch again to SHELTER, MEDICAL, or COMMUNICATION procedures (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 6: follow-up on shelter/medical/communication from Turn 5 (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Use chunks from at least two different hazard types. "
            "The topic field in each turn must accurately reflect the current hazard being discussed - "
            "not always the primary_topic. Adapt chunk selection across turns for realism."
        ),
    },
    "E": {
        "name": "Ambiguous / Non-Standalone Query",
        "turns": 6,
        "chunks_mode": "single_topic",
        "description": (
            "Turn 1: detailed, specific question about the main disaster topic (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: user follow-up 'What about children?' - ambiguous, references Turn 1 context (retrieval_required=true, query_type=clarification)\n"
            "Turn 3: user asks 'Is it safe?' - completely non-standalone, no explicit subject (retrieval_required=false, query_type=clarification)\n"
            "Turn 4: user asks 'How long should we wait?' - non-standalone, implicit reference (retrieval_required=false, query_type=clarification)\n"
            "Turn 5: user asks a slightly broader related question needing KB (retrieval_required=true, query_type=followup)\n"
            "Turn 6: user asks something unanswerable - specific local detail not in SOP (retrieval_required=false, query_type=unanswerable)"
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
            "Turn 5: question PARTIALLY supported - KB gives general info but not the specific detail asked (retrieval_required=true, query_type=followup) - assistant answers what KB supports and acknowledges the gap\n"
            "Turn 6: user rephrases Turn 4 with broader scope - now answerable from KB (retrieval_required=true, query_type=clarification)"
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
            "Turn 1: PREPAREDNESS - what to do before disaster strikes (retrieval_required=true, query_type=first_turn)\n"
            "Turn 2: WARNING/MONITORING - what signs indicate the disaster is approaching (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 3: follow-up on warning signs from Turn 2 (retrieval_required=false, query_type=followup)\n"
            "Turn 4: EVACUATION - when and how to evacuate (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 5: RESCUE - what to do if someone is trapped or injured (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 6: RECOVERY - how to safely return and recover after the event (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 7: follow-up on recovery from Turn 6 (retrieval_required=false, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: This episode must progress through disaster lifecycle stages: "
            "preparedness -> warning/monitoring -> evacuation -> rescue -> recovery. "
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
            "Turn 3: compare - ask about LANDSLIDE warning signs and how they differ from flood (retrieval_required=true, query_type=topic_shift)\n"
            "Turn 4: compare safe SHELTER behaviour for flood vs landslide (retrieval_required=true, query_type=clarification)\n"
            "Turn 5: ask which hazard requires FASTER evacuation and why - reason from prior turns only (retrieval_required=false, query_type=followup)\n"
            "Turn 6: practical comparison of supply kits for both hazards (retrieval_required=true, query_type=followup)"
        ),
        "extra_instructions": (
            "IMPORTANT: Use chunks from BOTH flood AND landslide topic groups. "
            "Turn 5 is a comparative reasoning turn - assistant must synthesise from conversation history only. "
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
            "Turn 4: OOD - user asks something completely unrelated to disaster response (e.g. recipe, sports, travel) (retrieval_required=false, query_type=OOD)\n"
            "Turn 5: OOD - user continues the off-topic tangent (retrieval_required=false, query_type=OOD)\n"
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
            "(e.g. 'If I should not stay, when exactly should I leave?') (retrieval_required=true, query_type=followup)\n"
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


def build_prompt(episode: Dict[str, Any]) -> str:
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
The dataset trains a PPO policy to decide WHEN to retrieve from a knowledge base vs use conversation memory.

## SCENARIO: {episode['scenario']} - {template['name']}
## PRIMARY TOPIC: {topic}
## AVAILABLE CHUNK IDs: {chunk_ids}

## KNOWLEDGE BASE (use ONLY these chunks for retrieval_required=true turns)
{chunk_block}

## TURN-BY-TURN STRUCTURE - follow exactly
{template['description']}
{extra_block}

## STRICT RULES
1. Resident speaks naturally - scared, confused, urgent, or curious depending on scenario.
2. Assistant is calm, concise (1-3 sentences), professional.
3. retrieval_required=true -> assistant answer MUST be grounded in KB chunks above.
4. retrieval_required=false -> assistant answers from conversation context ONLY, no new KB facts.
5. relevant_chunks: list of exact chunk_id strings used, or [] if not grounded in KB.
6. query_type: EXACTLY one of: first_turn | followup | topic_shift | clarification | OOD | unanswerable
7. retrieval_required logic:
   new topic / first factual need                 -> true
   follow-up answerable from prior assistant turn -> false
   ambiguous / clarification needing KB           -> true
   OOD (unrelated to disaster KB)                 -> false
   unanswerable (KB cannot answer)                -> false
8. Every turn: turn_id, query, assistant_answer, relevant_chunks, retrieval_required, topic, query_type
9. topic field reflects the CURRENT subject of that turn (may change per turn).

## OUTPUT - ONLY valid JSON, no markdown:
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
      "relevant_chunks": ["SOP_XXX"],
      "retrieval_required": true,
      "topic": "...",
      "query_type": "first_turn"
    }}
  ]
}}

Generate the complete {turn_count}-turn episode now. Ensure EXACTLY {turn_count} turns."""

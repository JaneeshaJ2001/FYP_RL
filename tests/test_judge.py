# test_judge.py  — run with: python test_judge.py
from rl_core.utils import JudgeChain, format_chunks_for_judge
from langchain_core.documents import Document

judge = JudgeChain()

# Case 1: good skip — prior summary has the answer
result = judge.invoke({
    "query": "Should we try to walk through the water to leave?",
    "conversation_summary": "User is trapped by flooding. Prior advice: never walk through floodwater due to currents and hidden debris. Evacuate to highest floor.",
    "action_taken": "skip",
    "retrieved_chunks": "",  # auto-filled to "No retrieval performed"
    "response": "Do NOT walk through floodwater. It is dangerous due to strong currents and hidden debris. Stay at the highest point and await rescue.",
})
print("Skip (good):", result)  # expect score 8-10

# Case 2: retrieve — good use of chunks
docs = [Document(page_content="During floods, boil all water before drinking. Avoid tap water that may be contaminated with sewage.")]
result = judge.invoke({
    "query": "Is tap water safe to drink after flooding?",
    "conversation_summary": "Village flooded. User asking about water safety.",
    "action_taken": "retrieve",
    "retrieved_chunks": format_chunks_for_judge(docs),
    "response": "No, tap water is not safe after flooding. Boil all water before drinking as it may be contaminated with sewage.",
})
print("Retrieve (good):", result)  # expect score 8-10

# Case 3: out-of-scope hallucination — should score low
result = judge.invoke({
    "query": "What stocks should I buy after the flood?",
    "conversation_summary": "Flood recovery context.",
    "action_taken": "skip",
    "retrieved_chunks": "",
    "response": "You should buy insurance stocks and construction companies as they benefit from disaster recovery.",
})
print("Out-of-scope hallucination:", result)  # expect score 0-3
from core.nodes import INITIAL_GROUNDED_SUMMARY
from rl_core.state_encoder import encode_state, _build_encoder_text

# Turn 1: no prior summary, no prior action
text1 = _build_encoder_text(INITIAL_GROUNDED_SUMMARY, "Water is rising near my house.", 1, -1)
print("Turn 1 input text:")
print(text1)
print()

obs1 = encode_state(INITIAL_GROUNDED_SUMMARY, "Water is rising near my house.", 1, -1)
print(f"Turn 1 obs shape: {obs1.shape}  dtype: {obs1.dtype}")
print()

# Turn 2: grounded summary exists, prev_action was RETRIEVE (1)
grounded = "Evacuation required immediately; move to higher ground per DMC SOP_164."
text2 = _build_encoder_text(grounded, "Should we take our documents?", 2, 1)
print("Turn 2 input text:")
print(text2)
print()

obs2 = encode_state(grounded, "Should we take our documents?", 2, 1)
print(f"Turn 2 obs shape: {obs2.shape}  dtype: {obs2.dtype}")

# Turn 3: prev_action was SKIP (0)
text3 = _build_encoder_text(grounded, "What about drinking water?", 3, 0)
print()
print("Turn 3 input text (prev SKIP):")
print(text3)

print("\nAll checks passed.")
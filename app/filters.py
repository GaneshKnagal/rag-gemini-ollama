\
PROFANITY_LIST = {"fuck","shit","bitch","bastard","asshole","dick","cunt"}
def contains_profanity(t:str)->bool: return any(w in t.lower() for w in PROFANITY_LIST)
def hallucination_gate(max_score: float, threshold: float):
    if max_score < threshold: return False, "I couldn’t find this information in the provided documents."
    return True, ""

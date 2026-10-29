"""Conservative French duration estimate for official and Faster batch attempts."""
import math
import re

def estimate_generation_steps(text):
    words = len(re.findall(r"\w+(?:['’\-]\w+)*", text))
    letters = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    pauses = len(re.findall(r"[,;:]", text))
    endings = len(re.findall(r"[.!?…]", text))
    return 8 + 0.2 * words + 0.7 * letters + 4 * digits + 2.2 * pauses + 3 * endings

def generation_step_budget(text, maximum=2048):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('Limite globale de génération invalide.')
    # The original estimate includes 10% slack, 24 pause frames and a 64-frame
    # floor. Allow 50% more than that entire budget, rounded up exactly,
    # then give every passage at least 300 steps, within the global ceiling.
    # Native individual retries are unchanged.
    estimated_budget = max(64, math.ceil(estimate_generation_steps(text) * 1.1 + 24))
    return min(maximum, max(300, (estimated_budget * 150 + 99) // 100))

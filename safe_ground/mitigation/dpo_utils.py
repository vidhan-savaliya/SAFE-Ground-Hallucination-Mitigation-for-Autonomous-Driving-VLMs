"""Mitigation Mode 3: DPO Preference Pair Construction.

Builds (prompt, chosen_response, rejected_response) pairs for Direct Preference
Optimization (DPO) to reduce hallucination rates during fine-tuning.
"""

from typing import List, Dict, Any
from safe_ground.data.schema import SampleData, Claim


def build_dpo_preference_pair(
    question: str,
    hallucinated_answer: str,
    corrected_answer: str,
    system_prompt: str = "You are an autonomous driving perception assistant.",
) -> Dict[str, str]:
    """Formats a single DPO preference record matching HuggingFace DPOTrainer schema."""
    return {
        "prompt": f"{system_prompt}\nUser: {question}\nAssistant:",
        "chosen": corrected_answer.strip(),
        "rejected": hallucinated_answer.strip(),
    }


def filter_pairs_by_severity_margin(
    candidate_pairs: List[Dict[str, Any]],
    min_severity_diff: float = 10.0,
) -> List[Dict[str, Any]]:
    """Filters DPO pairs to only keep samples where the rejected answer had significantly higher SWHS."""
    filtered = []
    for pair in candidate_pairs:
        swhs_rejected = pair.get("swhs_rejected", 0.0)
        swhs_chosen = pair.get("swhs_chosen", 0.0)
        margin = swhs_rejected - swhs_chosen
        if margin >= min_severity_diff:
            filtered.append(pair)
    return filtered


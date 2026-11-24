"""Mitigation Mode 1: Correction Re-prompter.

Formats verified sensor evidence and flagged errors into structured prompts
for VLM re-generation and answer correction.
"""

from typing import List, Dict, Any
from safe_ground.data.schema import Claim, Detection3D
from safe_ground.utils.constants import CLAIM_LABEL_CONTRADICTED


class CorrectionPromptGenerator:
    """Generates structured correction re-prompts from verifier outputs."""

    def __init__(self, template: str = None):
        self.template = template or self._default_template

    def _default_template(
        self,
        original_question: str,
        original_answer: str,
        contradicted_claims: List[Claim],
        omitted_detections: List[Detection3D],
    ) -> str:
        prompt_parts = [
            f"Original Driving Scene Question: {original_question}\n",
            f"Your Previous Response: {original_answer}\n",
            "A calibrated 3D perception cross-verifier analyzed your response and flagged the following safety-critical discrepancies:\n",
        ]

        # 1. Contradictions (Commissions / False Attributes)
        if contradicted_claims:
            prompt_parts.append("### Contradicted or Unsupported Claims (False Positives / Misattributions):")
            for c in contradicted_claims:
                prompt_parts.append(
                    f"- Claimed '{c.object_class}' at {c.bearing_sector} ({c.distance_bin} distance) "
                    f"with attribute '{c.attribute}' is contradicted by sensor evidence."
                )
            prompt_parts.append("")

        # 2. Omissions (Missed Objects)
        if omitted_detections:
            prompt_parts.append("### Safety-Critical Omissions (Objects you missed):")
            for d in omitted_detections:
                prompt_parts.append(
                    f"- Missed {d.class_name} at 3D position [x={d.x:.1f}m, y={d.y:.1f}m, z={d.z:.1f}m], "
                    f"confidence={d.confidence:.2f}."
                )
            prompt_parts.append("")

        prompt_parts.append(
            "Instruction: Please re-generate an accurate, safety-calibrated description of the scene that "
            "removes the contradicted claims and explicitly includes the missed critical entities."
        )

        return "\n".join(prompt_parts)

    def generate_prompt(
        self,
        question: str,
        answer: str,
        claims: List[Claim],
        claim_verification_labels: List[int],
        evidence_detections: List[Detection3D],
        omission_predictions: List[int],
    ) -> str:
        """Constructs the prompt given raw model predictions."""
        contradicted = [
            c for c, label in zip(claims, claim_verification_labels)
            if label == CLAIM_LABEL_CONTRADICTED
        ]
        omitted = [
            d for d, is_om in zip(evidence_detections, omission_predictions)
            if is_om == 1
        ]
        return self._default_template(question, answer, contradicted, omitted)


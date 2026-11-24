"""Mitigation Mode 2: Risk-Calibrated Abstention Policy."""

from typing import List, Dict, Any, Tuple
from safe_ground.data.schema import Claim, Detection3D
from safe_ground.eval.swhs import compute_object_severity
from safe_ground.utils.constants import CLAIM_LABEL_CONTRADICTED
from safe_ground.utils.geometry import sector_and_dist_to_coords


class RiskAbstentionPolicy:
    """Decides whether a VLM response is safe to pass downstream or should trigger an abstain warning."""

    def __init__(
        self,
        severity_threshold: float = 25.0,  # Max allowable severity before abstention
        vru_miss_threshold: int = 1,       # Abstain immediately if any VRU within 20m is omitted
    ):
        self.severity_threshold = severity_threshold
        self.vru_miss_threshold = vru_miss_threshold

    def evaluate(
        self,
        claims: List[Claim],
        claim_labels: List[int],
        detections: List[Detection3D],
        omission_labels: List[int],
    ) -> Tuple[bool, str, float]:
        """Returns:
          should_abstain: bool (True = unsafe / abstain)
          reason: str
          estimated_risk_score: float
        """
        # Calculate predicted severity of discrepancies
        total_risk = 0.0

        # Check contradicted claims
        for c, lbl in zip(claims, claim_labels):
            if lbl == CLAIM_LABEL_CONTRADICTED:
                x, y, z = sector_and_dist_to_coords(c.bearing_sector, c.distance_bin)
                sev = compute_object_severity(c.object_class, x, y, z)
                total_risk += sev

        # Check omitted evidence objects
        critical_vru_omissions = 0
        for d, is_om in zip(detections, omission_labels):
            if is_om == 1:
                sev = compute_object_severity(d.class_name, d.x, d.y, d.z, d.vx, d.vy, d.vz)
                total_risk += sev

                is_vru = d.class_name.lower() in {"pedestrian", "cyclist", "bicycle", "motorcycle"}
                dist = (d.x**2 + d.y**2)**0.5
                if is_vru and dist <= 20.0:
                    critical_vru_omissions += 1

        if critical_vru_omissions >= self.vru_miss_threshold:
            return (
                True,
                f"SAFETY_ABSTAIN: Omitted {critical_vru_omissions} nearby Vulnerable Road User(s) (<20m)!",
                total_risk,
            )

        if total_risk >= self.severity_threshold:
            return (
                True,
                f"SAFETY_ABSTAIN: Predicted discrepancy severity {total_risk:.1f} exceeds threshold {self.severity_threshold}!",
                total_risk,
            )

        return False, "PROCEED: Visual scene understanding confidence within safe parameters.", total_risk


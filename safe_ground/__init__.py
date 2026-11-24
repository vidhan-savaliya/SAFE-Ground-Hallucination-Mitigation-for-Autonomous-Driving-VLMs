"""SAFE-Ground: Severity-Aware, Omission-Aware Hallucination Detection & Mitigation."""

from safe_ground.models.safe_ground_model import SAFEGroundModel
from safe_ground.eval.swhs import SWHSCalculator
from safe_ground.data.claim_parser import ClaimParser
from safe_ground.data.schema import Claim, Detection3D, SampleData

__version__ = "0.1.0"

__all__ = [
    "SAFEGroundModel",
    "SWHSCalculator",
    "ClaimParser",
    "Claim",
    "Detection3D",
    "SampleData",
]

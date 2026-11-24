"""Driving-CHAIR: Instance- and Sentence-Level Hallucination Metric for Driving VLMs."""

from typing import List, Dict, Any, Set
import numpy as np
from safe_ground.data.schema import Claim


class DrivingCHAIR:
    """Computes CHAIR_i (instance) and CHAIR_s (sentence) adapted for driving domain."""

    def __init__(self, vru_classes: Set[str] = None):
        self.vru_classes = vru_classes or {"pedestrian", "cyclist", "bicycle", "motorcycle"}

    def evaluate_sample(
        self,
        claims: List[Claim],
        ground_truth_objects: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Evaluates a single sample.
        
        CHAIR_i = # hallucinated objects / # total objects mentioned
        CHAIR_s = 1 if sample contains >= 1 hallucinated object else 0
        """
        gt_classes = {
            str(gt.get("class_name", gt.get("class", ""))).lower()
            for gt in ground_truth_objects
        }

        total_claims = len(claims)
        if total_claims == 0:
            return {
                "chair_i": 0.0,
                "chair_s": 0.0,
                "chair_vru": 0.0,
                "num_claims": 0,
            }

        hallucinated_claims = 0
        vru_claims = 0
        vru_hallucinated = 0

        for c in claims:
            cls_lower = c.object_class.lower()
            is_hallucinated = (cls_lower not in gt_classes) or (c.ground_truth_status == "contradicted")

            if is_hallucinated:
                hallucinated_claims += 1

            if cls_lower in self.vru_classes:
                vru_claims += 1
                if is_hallucinated:
                    vru_hallucinated += 1

        chair_i = (hallucinated_claims / total_claims) * 100.0
        chair_s = 100.0 if hallucinated_claims > 0 else 0.0
        chair_vru = (vru_hallucinated / vru_claims * 100.0) if vru_claims > 0 else 0.0

        return {
            "chair_i": chair_i,
            "chair_s": chair_s,
            "chair_vru": chair_vru,
            "num_claims": total_claims,
            "num_hallucinated": hallucinated_claims,
        }

    def aggregate(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Averages across all evaluated samples."""
        if not results:
            return {"chair_i": 0.0, "chair_s": 0.0, "chair_vru": 0.0}

        chair_i_vals = [r["chair_i"] for r in results]
        chair_s_vals = [r["chair_s"] for r in results]
        chair_vru_vals = [r["chair_vru"] for r in results if r.get("chair_vru") is not None]

        return {
            "mean_chair_i": float(np.mean(chair_i_vals)),
            "mean_chair_s": float(np.mean(chair_s_vals)),
            "mean_chair_vru": float(np.mean(chair_vru_vals)) if chair_vru_vals else 0.0,
        }


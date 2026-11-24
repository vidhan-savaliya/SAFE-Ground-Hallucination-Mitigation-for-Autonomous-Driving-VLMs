"""Unified evaluation framework for SAFE-Ground (including FRS-SWHS and Evidential metrics)."""

from typing import List, Dict, Any, Optional
import numpy as np

from safe_ground.data.schema import SampleData, Claim, Detection3D
from safe_ground.eval.swhs import SWHSCalculator
from safe_ground.eval.driving_chair import DrivingCHAIR
from safe_ground.utils.constants import CLASS_SEVERITY_WEIGHTS


class SAFEGroundEvaluator:
    """Runs comprehensive benchmark evaluations across classical and PhD reachability metrics."""

    def __init__(
        self,
        swhs_calculator: Optional[SWHSCalculator] = None,
        chair_evaluator: Optional[DrivingCHAIR] = None,
    ):
        self.swhs_calc = swhs_calculator or SWHSCalculator()
        self.chair_eval = chair_evaluator or DrivingCHAIR()

    def evaluate_dataset(
        self,
        samples: List[SampleData],
        predicted_claims_list: List[List[Claim]],
    ) -> Dict[str, Any]:
        """Evaluates model predictions across full validation dataset."""
        sample_swhs = []
        sample_chair = []

        total_vru_omissions = 0
        total_vru_gt = 0
        total_near_omissions = 0
        total_near_gt = 0

        for sample, claims in zip(samples, predicted_claims_list):
            gt_objects = sample.ground_truth_boxes
            if not gt_objects and sample.detections:
                gt_objects = [
                    {"class_name": d.class_name, "x": d.x, "y": d.y, "z": d.z, "vx": d.vx, "vy": d.vy, "vz": d.vz}
                    for d in sample.detections
                ]

            # 1. SWHS (Classical and FRS Reachability)
            swhs_res = self.swhs_calc.compute_sample_swhs(claims, gt_objects, sample.detections)
            sample_swhs.append(swhs_res)

            # 2. Driving-CHAIR
            chair_res = self.chair_eval.evaluate_sample(claims, gt_objects)
            sample_chair.append(chair_res)

            # 3. Fine-grained Omission Tracking
            claimed_classes = {c.object_class.lower() for c in claims}
            for gt in gt_objects:
                cls_name = str(gt.get("class_name", "")).lower()
                dist = np.sqrt(gt.get("x", 0.0)**2 + gt.get("y", 0.0)**2)
                is_vru = cls_name in {"pedestrian", "cyclist", "bicycle", "motorcycle"}
                is_near = dist <= 15.0
                is_omitted = (cls_name not in claimed_classes) or gt.get("is_omitted", False)

                if is_vru:
                    total_vru_gt += 1
                    if is_omitted:
                        total_vru_omissions += 1

                if is_near:
                    total_near_gt += 1
                    if is_omitted:
                        total_near_omissions += 1

        # Aggregate Metrics
        swhs_agg = self.swhs_calc.aggregate_dataset(sample_swhs)
        chair_agg = self.chair_eval.aggregate(sample_chair)

        vru_omission_rate = (total_vru_omissions / total_vru_gt * 100.0) if total_vru_gt > 0 else 0.0
        near_omission_rate = (total_near_omissions / total_near_gt * 100.0) if total_near_gt > 0 else 0.0

        return {
            "num_evaluated_samples": len(samples),
            "SWHS": {
                "mean": swhs_agg["mean_swhs"],
                "p95": swhs_agg["p95_swhs"],
                "std": swhs_agg["std_swhs"],
                "com_component": swhs_agg["mean_swhs_com"],
                "om_component": swhs_agg["mean_swhs_om"],
                "attr_component": swhs_agg["mean_swhs_attr"],
                "mean_frs_swhs": swhs_agg["mean_frs_swhs"],
                "p95_frs_swhs": swhs_agg["p95_frs_swhs"],
            },
            "CHAIR": {
                "chair_i": chair_agg["mean_chair_i"],
                "chair_s": chair_agg["mean_chair_s"],
                "chair_vru": chair_agg["mean_chair_vru"],
            },
            "Omission_Rates": {
                "vru_omission_rate": vru_omission_rate,
                "near_omission_rate": near_omission_rate,
                "total_vru_objects": total_vru_gt,
                "total_near_objects": total_near_gt,
            },
        }

    @staticmethod
    def format_markdown_table(results: Dict[str, Any], model_name: str = "SAFE-Ground") -> str:
        """Generates publication-ready markdown evaluation table."""
        swhs = results.get("SWHS", {})
        chair = results.get("CHAIR", {})
        om = results.get("Omission_Rates", {})

        md = f"""### Benchmark Evaluation Summary: {model_name}
| Metric | Value | Description |
|:---|:---|:---|
| **SWHS (Total)** | **{swhs.get('mean', 0.0):.2f}** | Classical Severity-Weighted Hallucination Score |
| **FRS-SWHS (PhD)** | **{swhs.get('mean_frs_swhs', 0.0):.2f}** | Control-Theoretic Reachability Spatio-Temporal Score |
| **FRS-SWHS (95th Tail)** | {swhs.get('p95_frs_swhs', 0.0):.2f} | Worst-case 95th-percentile kinematic collision risk |
| SWHS Commission | {swhs.get('com_component', 0.0):.2f} | Weighted false discovery component |
| SWHS Omission | {swhs.get('om_component', 0.0):.2f} | Weighted critical miss component |
| **Driving-CHAIR_i** | {chair.get('chair_i', 0.0):.2f}% | Instance-level false object rate |
| **Driving-CHAIR_s** | {chair.get('chair_s', 0.0):.2f}% | Sentence-level hallucination rate |
| **CHAIR_VRU** | {chair.get('chair_vru', 0.0):.2f}% | Pedestrian/cyclist false positive rate |
| **VRU Omission Rate** | {om.get('vru_omission_rate', 0.0):.2f}% | Miss rate on vulnerable road users |
| **Near Omission Rate (<15m)** | {om.get('near_omission_rate', 0.0):.2f}% | Miss rate on nearby obstacles |
"""
        return md

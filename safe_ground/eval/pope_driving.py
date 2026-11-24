"""POPE-Driving: Polling-based Object Probing Evaluation adapted for Autonomous Driving."""

import random
from typing import List, Dict, Any, Set
import numpy as np


class POPEDrivingEvaluator:
    """Evaluates driving VLMs on binary existence queries ('Is there a <class> in the scene?').
    
    Implements 3 sampling splits:
    1. Random: randomly samples non-existent classes.
    2. Popular: polls frequently occurring driving objects (car, pedestrian, truck).
    3. Adversarial: polls classes co-occurring in driving scenes (e.g., cyclist when pedestrian is near).
    """

    DRIVING_CLASSES = [
        "car", "pedestrian", "truck", "bus", "cyclist", "motorcycle",
        "traffic_cone", "barrier", "traffic_light", "traffic_sign"
    ]

    POPULAR_CLASSES = ["car", "pedestrian", "truck", "traffic_light"]

    CO_OCCURRENCE_MAP = {
        "pedestrian": ["cyclist", "motorcycle", "traffic_cone"],
        "cyclist": ["pedestrian", "motorcycle", "bicycle"],
        "car": ["truck", "bus", "barrier"],
        "truck": ["trailer", "construction_vehicle", "bus"],
    }

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_polls(
        self,
        ground_truth_classes: Set[str],
        split: str = "adversarial",
        num_polls: int = 6,
    ) -> List[Dict[str, Any]]:
        """Generates balanced (yes/no) questions for a scene."""
        polls = []
        present = list(ground_truth_classes.intersection(self.DRIVING_CLASSES))
        absent = list(set(self.DRIVING_CLASSES) - ground_truth_classes)

        num_yes = num_polls // 2
        num_no = num_polls - num_yes

        # 1. Positive queries (Ground Truth = Yes)
        if present:
            chosen_yes = self.rng.choices(present, k=num_yes)
            for c in chosen_yes:
                polls.append({
                    "object_class": c,
                    "question": f"Is there a {c} in this driving scene?",
                    "ground_truth_answer": "yes",
                    "split": split,
                })

        # 2. Negative queries (Ground Truth = No)
        if absent:
            if split == "popular":
                candidates = [c for c in self.POPULAR_CLASSES if c in absent] or absent
            elif split == "adversarial":
                # Find adversarial candidates related to present objects
                adv_candidates = []
                for p in present:
                    adv_candidates.extend(self.CO_OCCURRENCE_MAP.get(p, []))
                candidates = [c for c in adv_candidates if c in absent] or absent
            else:  # random
                candidates = absent

            chosen_no = self.rng.choices(candidates, k=num_no)
            for c in chosen_no:
                polls.append({
                    "object_class": c,
                    "question": f"Is there a {c} in this driving scene?",
                    "ground_truth_answer": "no",
                    "split": split,
                })

        return polls

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """Calculates standard POPE metrics: Accuracy, Precision, Recall, F1, Yes-Ratio.
        
        predictions: list of dicts with 'predicted_answer' ('yes'/'no') and 'ground_truth_answer'.
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        total_yes_pred = 0

        for p in predictions:
            pred = p["predicted_answer"].strip().lower()
            gt = p["ground_truth_answer"].strip().lower()

            is_pred_yes = ("yes" in pred)
            is_gt_yes = ("yes" in gt)

            if is_pred_yes:
                total_yes_pred += 1

            if is_pred_yes and is_gt_yes:
                tp += 1
            elif is_pred_yes and not is_gt_yes:
                fp += 1
            elif not is_pred_yes and not is_gt_yes:
                tn += 1
            else:
                fn += 1

        total = len(predictions)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        yes_ratio = total_yes_pred / total if total > 0 else 0.0

        return {
            "pope_accuracy": accuracy * 100.0,
            "pope_precision": precision * 100.0,
            "pope_recall": recall * 100.0,
            "pope_f1": f1 * 100.0,
            "pope_yes_ratio": yes_ratio * 100.0,
        }


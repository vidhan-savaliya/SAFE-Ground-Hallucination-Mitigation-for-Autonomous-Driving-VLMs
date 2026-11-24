"""Synthetic hallucination injection pipeline for verifier training.

Corrupts clean DriveLM / nuScenes answers to produce:
1. Commission hallucinations (phantom objects)
2. Omission hallucinations (deleted real objects)
3. Attribute / spatial corruptions (wrong distance, wrong bearing, wrong state)
"""

import random
import copy
from typing import List, Tuple, Dict, Any
from safe_ground.data.schema import Claim, Detection3D
from safe_ground.utils.constants import (
    CLASS_NAME_TO_ID,
    BEARING_SECTORS,
    DISTANCE_BINS,
)


class HallucinationInjector:
    """Injects programmatic hallucinations into clean scene annotations."""

    def __init__(
        self,
        commission_prob: float = 0.35,
        omission_prob: float = 0.35,
        attribute_prob: float = 0.30,
        seed: int = 42,
    ):
        self.commission_prob = commission_prob
        self.omission_prob = omission_prob
        self.attribute_prob = attribute_prob
        self.rng = random.Random(seed)

        self.candidate_classes = [
            "pedestrian", "cyclist", "car", "truck", "traffic_cone", "barrier"
        ]
        self.bearing_sectors = BEARING_SECTORS
        self.distance_bins = list(DISTANCE_BINS.keys())
        self.attributes = ["moving", "parked", "crossing", "stationary", "braking"]

    def process_sample(
        self,
        clean_claims: List[Claim],
        detections: List[Detection3D],
    ) -> Tuple[List[Claim], List[Detection3D], Dict[str, Any]]:
        """Takes clean ground-truth claims and detections, injects corruptions, and labels both."""
        claims = [copy.deepcopy(c) for c in clean_claims]
        evidence = [copy.deepcopy(e) for e in detections]

        # By default, clean claims are marked 'supported'
        for c in claims:
            c.ground_truth_status = "supported"

        # By default, detections are considered 'mentioned' (not omitted)
        for e in evidence:
            e.is_omitted = False

        meta = {
            "num_commissions": 0,
            "num_omissions": 0,
            "num_attribute_errors": 0,
        }

        # 1. Attribute / Spatial Corruptions
        for c in claims:
            if self.rng.random() < self.attribute_prob:
                corruption_type = self.rng.choice(["distance", "bearing", "attribute"])
                if corruption_type == "distance":
                    other_dists = [d for d in self.distance_bins if d != c.distance_bin]
                    c.distance_bin = self.rng.choice(other_dists)
                elif corruption_type == "bearing":
                    other_bearings = [b for b in self.bearing_sectors if b != c.bearing_sector]
                    c.bearing_sector = self.rng.choice(other_bearings)
                else:
                    c.attribute = self.rng.choice(self.attributes)
                c.ground_truth_status = "contradicted"
                meta["num_attribute_errors"] += 1

        # 2. Commission Injections (Phantom Objects)
        if self.rng.random() < self.commission_prob or len(claims) == 0:
            num_phantoms = self.rng.randint(1, 2)
            for _ in range(num_phantoms):
                phantom = Claim(
                    object_class=self.rng.choice(self.candidate_classes),
                    bearing_sector=self.rng.choice(self.bearing_sectors),
                    distance_bin=self.rng.choice(self.distance_bins),
                    attribute=self.rng.choice(self.attributes),
                    camera_view="CAM_FRONT",
                    ground_truth_status="contradicted",
                )
                claims.append(phantom)
                meta["num_commissions"] += 1

        # 3. Omission Injections (Delete claims corresponding to real detections)
        if len(claims) > 1 and (self.rng.random() < self.omission_prob):
            # Pick a claim to remove
            removed_idx = self.rng.randrange(len(claims))
            removed = claims.pop(removed_idx)
            # Find closest matching detection in evidence and mark as omitted
            matched_evidence = self._find_matching_evidence(removed, evidence)
            if matched_evidence is not None:
                matched_evidence.is_omitted = True
                meta["num_omissions"] += 1

        # Re-check omission status for all evidence detections against remaining claims
        self._update_evidence_omission_labels(claims, evidence)

        return claims, evidence, meta

    def _find_matching_evidence(
        self, claim: Claim, evidence: List[Detection3D]
    ) -> Detection3D:
        """Finds evidence detection matching claim class."""
        for e in evidence:
            if e.class_name.lower() == claim.object_class.lower():
                return e
        return None

    def _update_evidence_omission_labels(
        self, claims: List[Claim], evidence: List[Detection3D]
    ):
        """Sets is_omitted=True for any evidence detection with no matching claim."""
        claimed_classes = {c.object_class.lower() for c in claims}
        for e in evidence:
            if e.class_name.lower() not in claimed_classes:
                e.is_omitted = True
            else:
                # If it wasn't already marked omitted by deliberate deletion
                if e.is_omitted is None:
                    e.is_omitted = False


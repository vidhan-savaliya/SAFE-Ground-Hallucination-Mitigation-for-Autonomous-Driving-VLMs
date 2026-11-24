"""PyTorch Dataset and Collator for SAFE-Ground training and evaluation."""

import math
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional

from safe_ground.data.schema import Claim, Detection3D, SampleData
from safe_ground.data.hallucination_injector import HallucinationInjector
from safe_ground.utils.constants import (
    CLASS_NAME_TO_ID,
    CLAIM_STATUS_TO_LABEL,
    CLAIM_LABEL_UNVERIFIABLE,
    PATH_WEIGHT_IN_PATH,
    PATH_WEIGHT_ADJACENT,
    PATH_WEIGHT_OFF_PATH,
)
from safe_ground.utils.geometry import sector_and_dist_to_coords, compute_distance_3d


class SAFEGroundDataset(Dataset):
    """Dataset producing batched claims, evidence detections, and severity metadata."""

    def __init__(
        self,
        samples: List[SampleData],
        inject_hallucinations: bool = False,
        injector_seed: int = 42,
    ):
        self.samples = samples
        self.inject_hallucinations = inject_hallucinations
        if inject_hallucinations:
            self.injector = HallucinationInjector(seed=injector_seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        claims = sample.claims
        detections = sample.detections

        if self.inject_hallucinations:
            claims, detections, _ = self.injector.process_sample(claims, detections)

        # 1. Process Claims
        claim_cls_list = []
        claim_targets_list = []
        claim_positions_list = []
        claim_distances_list = []
        claim_path_ids_list = []
        claim_vrel_list = []

        for c in claims:
            cid = CLASS_NAME_TO_ID.get(c.object_class.lower(), 9)
            claim_cls_list.append(cid)

            # Target status
            status_str = c.ground_truth_status or "unverifiable"
            status_id = CLAIM_STATUS_TO_LABEL.get(status_str, CLAIM_LABEL_UNVERIFIABLE)
            claim_targets_list.append(status_id)

            # Position approximation from sector + dist
            x, y, z = sector_and_dist_to_coords(c.bearing_sector, c.distance_bin)
            claim_positions_list.append([x, y, z])
            dist = compute_distance_3d(x, y, z)
            claim_distances_list.append(dist)

            # Lateral path relevance
            abs_y = abs(y)
            if abs_y <= 1.75:
                path_id = 0  # in-path
            elif abs_y <= 5.25:
                path_id = 1  # adjacent
            else:
                path_id = 2  # off-path
            claim_path_ids_list.append(path_id)
            claim_vrel_list.append(1.0)  # nominal

        # 2. Process Evidence Detections
        ev_class_ids = []
        ev_continuous = []
        ev_omission_targets = []
        ev_positions = []
        ev_distances = []
        ev_confidences = []
        ev_path_ids = []
        ev_vrel = []

        for d in detections:
            ev_class_ids.append(d.class_id)
            ev_continuous.append(d.to_feature_vector())
            ev_positions.append([d.x, d.y, d.z])
            ev_distances.append(compute_distance_3d(d.x, d.y, d.z))
            ev_confidences.append(d.confidence)

            # Omission target: 1 if omitted, 0 if mentioned
            is_om = 1.0 if d.is_omitted else 0.0
            ev_omission_targets.append(is_om)

            abs_y = abs(d.y)
            if abs_y <= 1.75:
                pid = 0
            elif abs_y <= 5.25:
                pid = 1
            else:
                pid = 2
            ev_path_ids.append(pid)
            v_norm = math.sqrt(d.vx * d.vx + d.vy * d.vy + d.vz * d.vz)
            ev_vrel.append(v_norm)

        return {
            "sample_id": sample.sample_id,
            "claim_classes": torch.tensor(claim_cls_list, dtype=torch.long),
            "claim_targets": torch.tensor(claim_targets_list, dtype=torch.long),
            "claim_positions": torch.tensor(claim_positions_list, dtype=torch.float32),
            "claim_distances": torch.tensor(claim_distances_list, dtype=torch.float32),
            "claim_path_ids": torch.tensor(claim_path_ids_list, dtype=torch.long),
            "claim_vrel": torch.tensor(claim_vrel_list, dtype=torch.float32),
            # Evidence
            "ev_class_ids": torch.tensor(ev_class_ids, dtype=torch.long),
            "ev_continuous": torch.tensor(ev_continuous, dtype=torch.float32),
            "ev_omission_targets": torch.tensor(ev_omission_targets, dtype=torch.float32),
            "ev_positions": torch.tensor(ev_positions, dtype=torch.float32),
            "ev_distances": torch.tensor(ev_distances, dtype=torch.float32),
            "ev_confidences": torch.tensor(ev_confidences, dtype=torch.float32),
            "ev_path_ids": torch.tensor(ev_path_ids, dtype=torch.long),
            "ev_vrel": torch.tensor(ev_vrel, dtype=torch.float32),
        }


def safe_ground_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates variable-length claims and evidence into padded tensors with boolean masks."""
    B = len(batch)

    # Maximum lengths in this batch
    max_claims = max(max(len(b["claim_classes"]), 1) for b in batch)
    max_ev = max(max(len(b["ev_class_ids"]), 1) for b in batch)

    # 1. Padded Claim Tensors
    claim_classes = torch.zeros(B, max_claims, dtype=torch.long)
    claim_targets = torch.zeros(B, max_claims, dtype=torch.long)
    claim_positions = torch.zeros(B, max_claims, 3, dtype=torch.float32)
    claim_distances = torch.zeros(B, max_claims, dtype=torch.float32)
    claim_path_ids = torch.zeros(B, max_claims, dtype=torch.long)
    claim_vrel = torch.zeros(B, max_claims, dtype=torch.float32)
    claim_mask = torch.zeros(B, max_claims, dtype=torch.bool)

    # 2. Padded Evidence Tensors
    ev_class_ids = torch.zeros(B, max_ev, dtype=torch.long)
    ev_continuous = torch.zeros(B, max_ev, 17, dtype=torch.float32)
    ev_omission_targets = torch.zeros(B, max_ev, dtype=torch.float32)
    ev_positions = torch.zeros(B, max_ev, 3, dtype=torch.float32)
    ev_distances = torch.zeros(B, max_ev, dtype=torch.float32)
    ev_confidences = torch.ones(B, max_ev, dtype=torch.float32)
    ev_path_ids = torch.zeros(B, max_ev, dtype=torch.long)
    ev_vrel = torch.zeros(B, max_ev, dtype=torch.float32)
    ev_mask = torch.zeros(B, max_ev, dtype=torch.bool)

    for i, b in enumerate(batch):
        nc = len(b["claim_classes"])
        if nc > 0:
            claim_classes[i, :nc] = b["claim_classes"]
            claim_targets[i, :nc] = b["claim_targets"]
            claim_positions[i, :nc] = b["claim_positions"]
            claim_distances[i, :nc] = b["claim_distances"]
            claim_path_ids[i, :nc] = b["claim_path_ids"]
            claim_vrel[i, :nc] = b["claim_vrel"]
            claim_mask[i, :nc] = True

        ne = len(b["ev_class_ids"])
        if ne > 0:
            ev_class_ids[i, :ne] = b["ev_class_ids"]
            ev_continuous[i, :ne] = b["ev_continuous"]
            ev_omission_targets[i, :ne] = b["ev_omission_targets"]
            ev_positions[i, :ne] = b["ev_positions"]
            ev_distances[i, :ne] = b["ev_distances"]
            ev_confidences[i, :ne] = b["ev_confidences"]
            ev_path_ids[i, :ne] = b["ev_path_ids"]
            ev_vrel[i, :ne] = b["ev_vrel"]
            ev_mask[i, :ne] = True

    return {
        "batch_size": B,
        "sample_ids": [b["sample_id"] for b in batch],
        # Claims
        "claim_classes": claim_classes,
        "claim_targets": claim_targets,
        "claim_positions": claim_positions,
        "claim_distances": claim_distances,
        "claim_path_ids": claim_path_ids,
        "claim_vrel": claim_vrel,
        "claim_mask": claim_mask,
        # Evidence
        "ev_class_ids": ev_class_ids,
        "ev_continuous": ev_continuous,
        "ev_omission_targets": ev_omission_targets,
        "ev_positions": ev_positions,
        "ev_distances": ev_distances,
        "ev_confidences": ev_confidences,
        "ev_path_ids": ev_path_ids,
        "ev_vrel": ev_vrel,
        "evidence_mask": ev_mask,
    }


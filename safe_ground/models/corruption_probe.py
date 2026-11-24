"""Corruption Awareness Probe & Evidential Vacuity Coupling Module.

Pillar 5 Extension (User Novelty):
1. Multi-camera Corruption Recognition Probe:
   - Predicts corruption class across 16 settings (Clean + 15 DriveBench corruptions)
   - Predicts per-camera degradation probability p_corr in [0, 1]^6
2. Evidential Adaptive Vacuity Coupling:
   - When corruption is detected on a camera view, evidence from that camera is discounted,
     gracefully elevating epistemic vacuity u in Dempster-Shafer Subjective Logic.
   - Claims are penalized less when the sensor itself is degraded.
3. Supports ablation toggle: enable_corruption_vacuity_coupling (True / False).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

NUM_CORRUPTIONS = 16  # Clean (0) + 15 Corruptions (1..15)


class CorruptionAwarenessProbe(nn.Module):
    """Probes multi-camera sensor representations for environmental/sensor corruptions."""

    def __init__(self, hidden_dim: int = 256, num_cameras: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_cameras = num_cameras

        # Per-camera degradation detector: hidden_dim -> 1 probability
        self.camera_degradation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Global scene corruption classifier: hidden_dim -> 16 classes
        self.scene_corruption_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_CORRUPTIONS)
        )

    def forward(self, evidence_tokens: torch.Tensor, cam_visibility: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        evidence_tokens : Tensor (B, M, D)
            Dense evidence representations from SensorEvidenceEncoder.
        cam_visibility : Tensor (B, M, 6), optional
            Binary camera visibility flags for each detection.

        Returns
        -------
        dict with:
            "camera_corruption_probs": Tensor (B, 6) in [0, 1]
            "scene_corruption_logits": Tensor (B, 16)
        """
        B, M, D = evidence_tokens.shape

        # Aggregate evidence tokens per camera view
        if cam_visibility is not None and cam_visibility.shape[-1] == 6:
            # Weighted average per camera
            vis_weights = cam_visibility / (cam_visibility.sum(dim=1, keepdim=True) + 1e-6)  # (B, M, 6)
            # cam_tokens: (B, 6, D)
            cam_tokens = torch.einsum('bmk,bmd->bkd', vis_weights, evidence_tokens)
        else:
            # Fallback mean pooling
            global_mean = evidence_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)
            cam_tokens = global_mean.repeat(1, self.num_cameras, 1)  # (B, 6, D)

        # 1. Per-camera corruption probabilities: (B, 6)
        camera_corruption_probs = self.camera_degradation_head(cam_tokens).squeeze(-1)

        # 2. Global scene corruption class logits: (B, 16)
        pooled_scene = cam_tokens.mean(dim=1)  # (B, D)
        scene_corruption_logits = self.scene_corruption_classifier(pooled_scene)

        return {
            "camera_corruption_probs": camera_corruption_probs,
            "scene_corruption_logits": scene_corruption_logits,
        }


class EvidentialSubjectiveLogicCoupler(nn.Module):
    """Couples corruption probe predictions to Dempster-Shafer Dirichlet priors.

    Elevates vacuity u when camera views are degraded, reducing false penalties.
    """

    def __init__(self, discount_factor_gamma: float = 0.85):
        super().__init__()
        self.gamma = discount_factor_gamma

    def forward(
        self,
        raw_evidence: torch.Tensor,
        camera_indices: torch.Tensor,
        camera_corruption_probs: torch.Tensor,
        enable_coupling: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        raw_evidence : Tensor (B, N, 3)
            Unmodulated evidence e_k >= 0 for 3 classes (Supported, Contradicted, Unverifiable).
        camera_indices : Tensor (B, N)
            Camera index in [0..5] associated with each claim.
        camera_corruption_probs : Tensor (B, 6)
            Degradation probability per camera from CorruptionAwarenessProbe.
        enable_coupling : bool
            If True, applies adaptive evidential discounting; if False, ablated (no link).

        Returns
        -------
        alpha : Tensor (B, N, 3) - Dirichlet parameters alpha_k >= 1.0
        belief : Tensor (B, N, 3) - Belief mass b_k in [0, 1]
        vacuity : Tensor (B, N) - Epistemic vacuity u in [0, 1]
        """
        B, N, K = raw_evidence.shape

        if enable_coupling and camera_corruption_probs is not None:
            # Gather corruption probability for each claim's camera view
            # camera_indices: (B, N) clamped to [0, 5]
            cam_idx_clamped = camera_indices.clamp(0, 5).long()
            # claim_p_corr: (B, N, 1)
            claim_p_corr = torch.gather(camera_corruption_probs, 1, cam_idx_clamped).unsqueeze(-1)

            # Discount evidence: e_tilde = e * (1 - gamma * p_corr)
            discount = (1.0 - self.gamma * claim_p_corr).clamp(min=0.05, max=1.0)
            modulated_evidence = raw_evidence * discount
        else:
            modulated_evidence = raw_evidence

        # Dirichlet parameters: alpha_k = e_k + 1.0 >= 1.0
        alpha = modulated_evidence + 1.0

        # Dirichlet strength S = sum(alpha_k)
        S = alpha.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Belief: b_k = e_k / S
        belief = modulated_evidence / (S + 1e-8)

        # Epistemic vacuity: u = K / S
        vacuity = (float(K) / (S + 1e-8)).squeeze(-1)  # (B, N)

        return alpha, belief, vacuity


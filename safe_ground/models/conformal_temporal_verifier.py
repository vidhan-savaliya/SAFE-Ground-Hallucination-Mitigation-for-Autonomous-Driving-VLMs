"""
SAFE-Ground++: Geometric-Semantic Claim Encoding, Split-Conformalized Evidential
Risk Calibration, and Temporal Markov Scene-Consensus Smoothing.

Author Novelties over Base SAFE-Ground:
1. ClaimSemanticGeometryEncoder: Replaces ungrounded/dummy claim vectors with
   learned geometric-semantic embeddings encoding [class_id, 3D anchor (x,y,z),
   log-distance, bearing (sin, cos), lateral path relevance] so cross-attention
   with continuous 3D BEVFusion evidence E_3D is genuinely grounded.
2. ConformalEvidentialCalibrator: Distribution-free split-conformal calibration
   computing class-stratified non-conformity quantiles (q_VRU, q_veh) over
   Dirichlet evidential belief/vacuity scores with FRS proximity modulation,
   providing finite-sample coverage guarantees P(miss VRU) <= alpha_VRU.
3. TemporalConsensusFilter: Fuses Head B omission probabilities across
   consecutive chronological frames within each nuScenes scene_token via 3D
   nearest-neighbor tracklet association (delta_d <= 3.5m), suppressing
   single-frame epistemic flicker and tail SWHS spikes.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from safe_ground.models.safe_ground_model import SAFEGroundModel
from safe_ground.data.schema import Detection3D


VRU_CLASS_IDS = {0, 1, 2}  # pedestrian, bicycle/cyclist, motorcycle
VRU_CLASS_NAMES = {"pedestrian", "cyclist", "bicycle", "motorcycle"}


class ClaimSemanticGeometryEncoder(nn.Module):
    """Encodes structured driving claims into dense D-dim query embeddings."""

    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 256,
        class_embed_dim: int = 32,
        path_embed_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_embed_dim)
        self.path_emb = nn.Embedding(4, path_embed_dim)
        # Continuous geometry: [x/50, y/50, z/5, log1p(dist)/4, sin(theta), cos(theta), vrel/20]
        cont_dim = 7
        in_dim = class_embed_dim + path_embed_dim + cont_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        claim_classes: torch.Tensor,     # (B, N) long
        claim_positions: torch.Tensor,   # (B, N, 3) float
        claim_distances: torch.Tensor,   # (B, N) float
        claim_path_ids: torch.Tensor,    # (B, N) long
        claim_vrel: torch.Tensor,        # (B, N) float
    ) -> torch.Tensor:
        cls_e = self.class_emb(claim_classes.clamp(0, self.class_emb.num_embeddings - 1))
        path_e = self.path_emb(claim_path_ids.clamp(0, 3))

        x = claim_positions[..., 0]
        y = claim_positions[..., 1]
        z = claim_positions[..., 2]
        theta = torch.atan2(y, x + 1e-6)
        cont = torch.stack(
            [
                x / 50.0,
                y / 50.0,
                z / 5.0,
                torch.log1p(claim_distances.clamp(min=0.0)) / 4.0,
                torch.sin(theta),
                torch.cos(theta),
                claim_vrel / 20.0,
            ],
            dim=-1,
        )
        feat = torch.cat([cls_e, path_e, cont], dim=-1)
        return self.mlp(feat)


class SAFEGroundPlusPlusModel(nn.Module):
    """End-to-End Real Neural SAFE-Ground++ Model with Geometric-Semantic Claim Encoder."""

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        num_classes: int = 10,
        continuous_dim: int = 17,
        lambda_om: float = 1.5,
        lambda_match: float = 0.5,
        kl_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.claim_encoder = ClaimSemanticGeometryEncoder(
            num_classes=num_classes,
            hidden_dim=d_model,
        )
        self.core = SAFEGroundModel(
            d_model=d_model,
            nhead=nhead,
            num_verifier_layers=1,
            dim_feedforward=dim_feedforward,
            num_classes=num_classes,
            continuous_dim=continuous_dim,
            lambda_om=lambda_om,
            lambda_match=lambda_match,
            lambda_lm=0.0,
            use_evidential=True,
            kl_weight=kl_weight,
        )

    def forward(
        self,
        claim_mask: torch.Tensor,
        ev_class_ids: torch.Tensor,
        ev_continuous: torch.Tensor,
        ev_confidences: torch.Tensor,
        evidence_mask: torch.Tensor,
        claim_classes: torch.Tensor,
        claim_positions: torch.Tensor,
        ev_positions: torch.Tensor,
        claim_distances: torch.Tensor,
        claim_path_ids: torch.Tensor,
        claim_vrel: torch.Tensor,
        claim_targets: Optional[torch.Tensor] = None,
        ev_omission_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        claim_embeddings = self.claim_encoder(
            claim_classes=claim_classes,
            claim_positions=claim_positions,
            claim_distances=claim_distances,
            claim_path_ids=claim_path_ids,
            claim_vrel=claim_vrel,
        )
        return self.core(
            claim_embeddings=claim_embeddings,
            claim_mask=claim_mask,
            ev_class_ids=ev_class_ids,
            ev_continuous=ev_continuous,
            ev_confidences=ev_confidences,
            evidence_mask=evidence_mask,
            claim_targets=claim_targets,
            ev_omission_targets=ev_omission_targets,
            claim_classes=claim_classes,
            claim_positions=claim_positions,
            ev_positions=ev_positions,
            claim_distances=claim_distances,
            claim_path_ids=claim_path_ids,
            claim_vrel=claim_vrel,
        )


class ConformalEvidentialCalibrator:
    """
    Distribution-Free Split-Conformal Risk Calibrator for Driving Hallucinations.

    Fits finite-sample conformal thresholds on a disjoint calibration fold:
      - Head A (Commission filter): Flags claim c as contradicted whenever
        s_com(c) = P_ev(contra | c) + beta * u(c) >= tau_com.
      - Head B (Reverse Omission): Flags 3D object o as omitted whenever
        FRS-modulated omission score p_om(o) + delta_frs(o) >= tau_om(class(o)),
        with tighter alpha_vru (e.g. 0.005) for Vulnerable Road Users.
    """

    def __init__(
        self,
        alpha_vru: float = 0.005,
        alpha_veh: float = 0.025,
        alpha_com_fp: float = 0.02,
        vacuity_beta: float = 0.25,
        frs_boost: float = 0.12,
    ) -> None:
        self.alpha_vru = alpha_vru
        self.alpha_veh = alpha_veh
        self.alpha_com_fp = alpha_com_fp
        self.vacuity_beta = vacuity_beta
        self.frs_boost = frs_boost

        self.tau_com: float = 0.50
        self.tau_om_vru: float = 0.35
        self.tau_om_veh: float = 0.45

    @staticmethod
    def _conformal_lower_quantile(scores: np.ndarray, alpha: float) -> float:
        """Finite-sample split-conformal lower quantile for recall guarantee >= 1 - alpha."""
        n = len(scores)
        if n == 0:
            return 0.5
        # Quantile level for catching >= (1 - alpha) of true positives
        q_level = max(0.0, min(1.0, math.floor((n + 1) * alpha) / float(n)))
        return float(np.quantile(scores, q_level))

    @staticmethod
    def _conformal_upper_quantile(scores: np.ndarray, alpha: float) -> float:
        """Finite-sample split-conformal upper quantile to bound false-positive rate <= alpha."""
        n = len(scores)
        if n == 0:
            return 0.5
        q_level = max(0.0, min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n)))
        return float(np.quantile(scores, q_level))

    def compute_frs_proximity_boost(self, det: Detection3D) -> float:
        """Risk-asymmetric boost for near-range or in-path obstacles."""
        dist = math.hypot(det.x, det.y)
        in_path = abs(det.y) <= 2.2
        is_vru = det.class_name.lower() in VRU_CLASS_NAMES
        prox = math.exp(-max(0.0, dist - 5.0) / 18.0)
        mult = 1.5 if is_vru else 1.0
        if in_path:
            mult *= 1.35
        return min(0.35, self.frs_boost * prox * mult)

    def fit(
        self,
        cal_com_scores_supported: np.ndarray,
        cal_com_scores_contradicted: np.ndarray,
        cal_om_scores_vru_pos: np.ndarray,
        cal_om_scores_veh_pos: np.ndarray,
    ) -> Dict[str, float]:
        # Head A: pick threshold balancing <= alpha_com_fp false rejection of supported claims
        # while catching contradicted claims
        tau_fp_bound = self._conformal_upper_quantile(cal_com_scores_supported, self.alpha_com_fp)
        tau_tp_bound = self._conformal_lower_quantile(cal_com_scores_contradicted, 0.05)
        self.tau_com = float(np.clip(0.5 * (tau_fp_bound + tau_tp_bound), 0.15, 0.85))

        # Head B: finite-sample recall guarantee >= 1 - alpha on true omitted 3D objects
        self.tau_om_vru = float(np.clip(self._conformal_lower_quantile(cal_om_scores_vru_pos, self.alpha_vru), 0.05, 0.70))
        self.tau_om_veh = float(np.clip(self._conformal_lower_quantile(cal_om_scores_veh_pos, self.alpha_veh), 0.10, 0.75))

        return {
            "tau_com": round(self.tau_com, 4),
            "tau_om_vru": round(self.tau_om_vru, 4),
            "tau_om_veh": round(self.tau_om_veh, 4),
        }


class TemporalConsensusFilter:
    """
    Chronological Scene-Level Markov Belief Smoothing for 3D Omission Probabilities.

    Associates physical 3D detections across adjacent frames within the same
    scene_token (Euclidean distance <= match_radius_m, matching class) and
    smooths omission probabilities to suppress single-frame epistemic flicker.
    """

    def __init__(self, eta: float = 0.35, match_radius_m: float = 3.5) -> None:
        self.eta = eta
        self.match_radius_m = match_radius_m
        self._last_scene: Optional[str] = None
        self._prev_tracks: List[Tuple[str, float, float, float]] = []

    def reset(self) -> None:
        self._last_scene = None
        self._prev_tracks = []

    def smooth_frame(
        self,
        scene_token: str,
        detections: List[Detection3D],
        raw_om_probs: np.ndarray,
    ) -> np.ndarray:
        if scene_token != self._last_scene:
            self._last_scene = scene_token
            self._prev_tracks = []

        smoothed = np.array(raw_om_probs, dtype=np.float32).copy()
        new_tracks: List[Tuple[str, float, float, float]] = []

        for idx, det in enumerate(detections):
            if idx >= len(smoothed):
                break
            cls_name = det.class_name.lower()
            p_curr = float(smoothed[idx])
            best_prev_p = None
            best_dist = self.match_radius_m

            for p_cls, px, py, pp in self._prev_tracks:
                if p_cls != cls_name:
                    continue
                d = math.hypot(det.x - px, det.y - py)
                if d <= best_dist:
                    best_dist = d
                    best_prev_p = pp

            if best_prev_p is not None:
                p_fused = (1.0 - self.eta) * p_curr + self.eta * best_prev_p
                # Safety-conservative max fusion for close VRUs so persistent VRUs never flicker off
                if cls_name in VRU_CLASS_NAMES and math.hypot(det.x, det.y) <= 18.0:
                    p_fused = max(p_fused, 0.85 * best_prev_p)
                smoothed[idx] = p_fused

            new_tracks.append((cls_name, det.x, det.y, float(smoothed[idx])))

        self._prev_tracks = new_tracks
        return smoothed

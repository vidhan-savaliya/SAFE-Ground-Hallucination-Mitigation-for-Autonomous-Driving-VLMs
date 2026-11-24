"""Top-level SAFE-Ground Model wrapping Evidence Encoder, Evidential Verifier, FRS-SWHS, and Loss."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from safe_ground.models.verifier import (
    SensorEvidenceEncoder,
    EvidenceCrossAttentionVerifier,
    ClaimEmbeddingExtractor,
)
from safe_ground.models.evidential import (
    EvidentialCrossAttentionHead,
    EvidentialVerificationLoss,
)
from safe_ground.losses import SAFEGroundTotalLoss, SeverityWeightComputer
from safe_ground.utils.reachability import KinematicReachabilityComputer
from safe_ground.mitigation.cbf_shield import NeuroSymbolicSafetyShield


class SAFEGroundModel(nn.Module):
    """End-to-End PhD-Grade SAFE-Ground Architecture:
    1. 3D Sensor Evidence Projection
    2. Confidence-Gated Cross-Attention Verifier
    3. Evidential Subjective Logic Head (Belief, Disbelief, Epistemic Vacuity)
    4. Kinematic FRS Reachability & CBF Safety Shield
    """

    def __init__(
        self,
        d_model: int = 3584,  # Qwen2.5-VL-7B hidden dimension (use 256 for test/debug)
        nhead: int = 8,
        num_verifier_layers: int = 1,
        dim_feedforward: int = 2048,
        num_classes: int = 10,
        continuous_dim: int = 17,
        lambda_om: float = 1.0,
        lambda_match: float = 0.5,
        lambda_lm: float = 0.0,
        use_evidential: bool = True,
        kl_weight: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_evidential = use_evidential

        # 1. Evidence Encoder (3D bounding box features -> LLM dim tokens)
        self.evidence_encoder = SensorEvidenceEncoder(
            num_classes=num_classes,
            hidden_dim=d_model,
            num_continuous=continuous_dim,
        )

        # 2. Claim Embedding Extractor
        self.claim_extractor = ClaimEmbeddingExtractor()

        # 3. Evidence-Cross-Attention Verifier Block
        self.verifier = EvidenceCrossAttentionVerifier(
            hidden_dim=d_model,
            num_heads=nhead,
            ffn_dim=dim_feedforward,
        )

        # 4. Pillar II: Evidential Head (Dirichlet distribution over Subjective Logic)
        if use_evidential:
            self.evidential_head = EvidentialCrossAttentionHead(hidden_dim=d_model, num_classes=3)
            self.evidential_loss_fn = EvidentialVerificationLoss(kl_weight=kl_weight)

        # 5. Pillar I & IV: Reachability Computer & CBF Safety Shield
        self.reachability_computer = KinematicReachabilityComputer()
        self.cbf_shield = NeuroSymbolicSafetyShield()

        # 6. Base Multi-Task Losses and Severity Computer
        self.total_loss_fn = SAFEGroundTotalLoss(
            lambda_om=lambda_om,
            lambda_match=lambda_match,
            lambda_lm=lambda_lm,
        )
        self.severity_computer = SeverityWeightComputer()

    def forward(
        self,
        # Claims
        claim_embeddings: torch.Tensor,       # (B, N, D)
        claim_mask: torch.Tensor,             # (B, N) bool
        # Evidence
        ev_class_ids: torch.Tensor,           # (B, M)
        ev_continuous: torch.Tensor,          # (B, M, 17)
        ev_confidences: torch.Tensor,         # (B, M)
        evidence_mask: torch.Tensor,          # (B, M) bool
        # Targets for loss calculation (optional during pure inference)
        claim_targets: Optional[torch.Tensor] = None,         # (B, N)
        ev_omission_targets: Optional[torch.Tensor] = None,   # (B, M)
        # Spatial metadata for Hungarian matching & reachability
        claim_classes: Optional[torch.Tensor] = None,         # (B, N)
        claim_positions: Optional[torch.Tensor] = None,       # (B, N, 3)
        ev_positions: Optional[torch.Tensor] = None,          # (B, M, 3)
        claim_distances: Optional[torch.Tensor] = None,       # (B, N)
        claim_path_ids: Optional[torch.Tensor] = None,        # (B, N)
        claim_vrel: Optional[torch.Tensor] = None,            # (B, N)
        lm_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Runs forward pass through evidence encoder, cross-attention verifier, and loss."""
        # 1. Encode 3D detections into evidence tokens
        evidence_tokens = self.evidence_encoder(
            class_ids=ev_class_ids,
            continuous_features=ev_continuous,
        )  # (B, M, D)

        # 2. Run cross-attention verifier
        verifier_out = self.verifier(
            claim_embeddings=claim_embeddings,
            evidence_tokens=evidence_tokens,
            evidence_confidence=ev_confidences,
            claim_mask=claim_mask,
            evidence_mask=evidence_mask,
        )
        claim_logits = verifier_out["claim_logits"]
        omission_logits = verifier_out["omission_logits"]

        outputs = {
            "claim_logits": claim_logits,         # (B, N, 3)
            "omission_logits": omission_logits,   # (B, M, 1)
            "evidence_tokens": evidence_tokens,   # (B, M, D)
        }

        # 3. Evidential Subjective Logic Inference
        if self.use_evidential:
            evidential_res = self.evidential_head(claim_embeddings)
            outputs["alpha"] = evidential_res["alpha"]
            outputs["beliefs"] = evidential_res["beliefs"]        # (B, N, 3) (supported, contradicted, unverifiable)
            outputs["vacuity"] = evidential_res["vacuity"]        # (B, N, 1) (epistemic sensor ignorance)
            outputs["expected_prob"] = evidential_res["expected_prob"]

        # 4. Compute loss if ground-truth targets are provided
        if claim_targets is not None and ev_omission_targets is not None:
            # Base loss dictionary
            losses = self.total_loss_fn(
                claim_logits=claim_logits,
                claim_targets=claim_targets,
                claim_mask=claim_mask,
                omission_logits=omission_logits,
                omission_targets=ev_omission_targets,
                evidence_mask=evidence_mask,
                class_ids=claim_classes if claim_classes is not None else torch.zeros_like(claim_targets),
                distances=claim_distances if claim_distances is not None else torch.ones_like(claim_targets).float() * 20.0,
                path_ids=claim_path_ids if claim_path_ids is not None else torch.zeros_like(claim_targets),
                v_rel_norms=claim_vrel if claim_vrel is not None else torch.ones_like(claim_targets).float(),
                pred_classes=claim_classes if claim_classes is not None else torch.zeros_like(claim_targets),
                gt_classes=ev_class_ids,
                pred_positions=claim_positions if claim_positions is not None else torch.zeros(claim_embeddings.shape[0], claim_embeddings.shape[1], 3, device=claim_embeddings.device),
                gt_positions=ev_positions if ev_positions is not None else torch.zeros(evidence_tokens.shape[0], evidence_tokens.shape[1], 3, device=evidence_tokens.device),
                pred_embeddings=claim_embeddings,
                gt_embeddings=evidence_tokens,
                lm_loss=lm_loss,
            )

            # Evidential Digamma Loss + Vacuity Attenuation
            if self.use_evidential:
                weights = self.severity_computer(
                    class_ids=claim_classes if claim_classes is not None else torch.zeros_like(claim_targets),
                    distances=claim_distances if claim_distances is not None else torch.ones_like(claim_targets).float() * 20.0,
                    path_ids=claim_path_ids if claim_path_ids is not None else torch.zeros_like(claim_targets),
                    v_rel_norms=claim_vrel if claim_vrel is not None else torch.ones_like(claim_targets).float(),
                )
                l_evidential = self.evidential_loss_fn(
                    alpha=outputs["alpha"],
                    targets=claim_targets,
                    severity_weights=weights,
                    mask=claim_mask,
                )
                losses["evidential_verification"] = l_evidential
                # Blend with total loss
                losses["total"] = losses["total"] + 1.2 * l_evidential

            outputs["losses"] = losses
            outputs["loss"] = losses["total"]

        return outputs

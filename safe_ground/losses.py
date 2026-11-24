"""
SAFE-Ground — Loss functions for the Evidence-Cross-Attention Verifier.

Modules
-------
SeverityWeightComputer
    Computes per-object severity weights using the fixed formula
    ``w(o) = λ_cls × f_d × λ_path × f_v``.

SeverityWeightedVerificationLoss
    Severity-weighted cross-entropy for the 3-class claim verifier (Head A).

SeverityWeightedOmissionLoss
    Severity-weighted binary cross-entropy for omission detection (Head B).

HungarianMatchingLoss
    Bipartite matching loss between predicted and ground-truth detections
    using ``scipy.optimize.linear_sum_assignment``.

SAFEGroundTotalLoss
    Aggregates all sub-losses with configurable coefficients.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# scipy is only needed by HungarianMatchingLoss — import lazily so that
# the rest of the module works even if scipy is missing.
try:
    from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Severity Weight Computer
# ---------------------------------------------------------------------------

# Default class-name → severity multiplier mapping
_DEFAULT_CLASS_SEVERITY: Dict[str, float] = {
    "pedestrian": 3.0,
    "cyclist": 3.0,
    "vehicle": 2.0,
    "traffic_sign": 1.5,
    "cone": 1.5,
    "static": 1.0,
}

# Integer-indexed default (same order as above, for convenience)
_DEFAULT_CLASS_SEVERITY_BY_ID: Dict[int, float] = {
    0: 3.0,   # pedestrian
    1: 3.0,   # cyclist
    2: 2.0,   # vehicle
    3: 1.5,   # traffic_sign
    4: 1.5,   # cone
    5: 1.0,   # static
}

_DEFAULT_PATH_SEVERITY: Dict[str, float] = {
    "in_path": 2.0,
    "adjacent": 1.5,
    "off_path": 1.0,
}

_DEFAULT_PATH_SEVERITY_BY_ID: Dict[int, float] = {
    0: 2.0,   # in_path
    1: 1.5,   # adjacent
    2: 1.0,   # off_path
}


class SeverityWeightComputer(nn.Module):
    """Compute per-object severity weights (deterministic, NOT learned).

    .. math::

        w(o) = \\lambda_{\\text{cls}}(\\text{class})
               \\times f_d(\\text{distance})
               \\times \\lambda_{\\text{path}}(\\text{path\\_rel})
               \\times f_v(v_{\\text{rel}})

    where:

    * :math:`f_d(d) = \\min\\bigl(d_{\\max} / (d + \\varepsilon),\\; w_{\\max}\\bigr)`
    * :math:`f_v(v) = 1 + 0.1 \\, \\|v_{\\text{rel}}\\|`

    Parameters
    ----------
    class_severity : dict[int, float] | None
        Mapping from integer class ID to :math:`\\lambda_{\\text{cls}}`.
        Defaults to the SAFE-Ground specification.
    path_severity : dict[int, float] | None
        Mapping from integer path-relation ID to :math:`\\lambda_{\\text{path}}`.
    d_max : float
        Distance normalisation constant.
    eps : float
        Epsilon added to distance to avoid division by zero.
    w_max : float
        Maximum distance weight.
    v_scale : float
        Velocity scaling factor (default 0.1).
    default_class_weight : float
        Fallback :math:`\\lambda_{\\text{cls}}` for unknown classes.
    default_path_weight : float
        Fallback :math:`\\lambda_{\\text{path}}` for unknown path relations.
    """

    def __init__(
        self,
        class_severity: Optional[Dict[int, float]] = None,
        path_severity: Optional[Dict[int, float]] = None,
        d_max: float = 50.0,
        eps: float = 1.0,
        w_max: float = 10.0,
        v_scale: float = 0.1,
        default_class_weight: float = 1.0,
        default_path_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.class_severity = class_severity or _DEFAULT_CLASS_SEVERITY_BY_ID
        self.path_severity = path_severity or _DEFAULT_PATH_SEVERITY_BY_ID
        self.d_max = d_max
        self.eps = eps
        self.w_max = w_max
        self.v_scale = v_scale
        self.default_class_weight = default_class_weight
        self.default_path_weight = default_path_weight

    @torch.no_grad()
    def forward(
        self,
        class_ids: torch.Tensor,
        distances: torch.Tensor,
        path_ids: torch.Tensor,
        v_rel_norms: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        class_ids : LongTensor ``(B, K)``
            Integer class IDs for each object.
        distances : Tensor ``(B, K)``
            Euclidean distance of each object from the ego vehicle.
        path_ids : LongTensor ``(B, K)``
            Integer path-relation IDs (0=in_path, 1=adjacent, 2=off_path).
        v_rel_norms : Tensor ``(B, K)``
            Norm of relative velocity for each object.

        Returns
        -------
        Tensor ``(B, K)``
            Per-object severity weights.
        """
        device = distances.device

        # λ_cls
        lam_cls = torch.tensor(
            [[self.class_severity.get(int(c), self.default_class_weight) for c in row]
             for row in class_ids.cpu()],
            dtype=torch.float32,
            device=device,
        )

        # f_d
        f_d = torch.clamp(self.d_max / (distances + self.eps), max=self.w_max)

        # λ_path
        lam_path = torch.tensor(
            [[self.path_severity.get(int(p), self.default_path_weight) for p in row]
             for row in path_ids.cpu()],
            dtype=torch.float32,
            device=device,
        )

        # f_v
        f_v = 1.0 + self.v_scale * v_rel_norms

        return lam_cls * f_d * lam_path * f_v


# ---------------------------------------------------------------------------
# Severity-Weighted Verification Loss  (Head A — 3-class CE)
# ---------------------------------------------------------------------------

class SeverityWeightedVerificationLoss(nn.Module):
    """Per-claim severity-weighted cross-entropy loss.

    Parameters
    ----------
    num_classes : int
        Number of verification classes (default 3).
    label_smoothing : float
        Label smoothing passed to ``F.cross_entropy``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor ``(B, N, C)``
            Raw class logits from Head A.
        targets : LongTensor ``(B, N)``
            Ground-truth class indices in ``{0, …, C-1}``.
        weights : Tensor ``(B, N)``
            Per-claim severity weights.
        mask : BoolTensor ``(B, N)``, optional
            ``True`` for valid claims.  If *None*, all positions are valid.

        Returns
        -------
        Scalar loss tensor.
        """
        B, N, C = logits.shape

        # Flatten to (B*N, C) / (B*N,)
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)
        weights_flat = weights.reshape(-1)

        # Per-element CE (unreduced)
        ce = F.cross_entropy(
            logits_flat,
            targets_flat,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )  # (B*N,)

        weighted_ce = ce * weights_flat

        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            weighted_ce = weighted_ce * mask_flat
            denom = mask_flat.sum().clamp(min=1.0)
        else:
            denom = torch.tensor(float(B * N), device=logits.device)

        return weighted_ce.sum() / denom


# ---------------------------------------------------------------------------
# Severity-Weighted Omission Loss  (Head B — binary BCE)
# ---------------------------------------------------------------------------

class SeverityWeightedOmissionLoss(nn.Module):
    """Per-evidence severity-weighted binary cross-entropy loss."""

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor ``(B, M, 1)`` or ``(B, M)``
            Raw omission logits from Head B.
        targets : FloatTensor ``(B, M)``
            Binary ground-truth (0 = mentioned, 1 = omitted).
        weights : Tensor ``(B, M)``
            Per-evidence severity weights.
        mask : BoolTensor ``(B, M)``, optional
            ``True`` for valid evidence positions.

        Returns
        -------
        Scalar loss tensor.
        """
        logits = logits.squeeze(-1)  # → (B, M)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )  # (B, M)

        weighted_bce = bce * weights

        if mask is not None:
            weighted_bce = weighted_bce * mask.float()
            denom = mask.float().sum().clamp(min=1.0)
        else:
            denom = torch.tensor(float(logits.numel()), device=logits.device)

        return weighted_bce.sum() / denom


# ---------------------------------------------------------------------------
# Hungarian Matching Loss
# ---------------------------------------------------------------------------

class HungarianMatchingLoss(nn.Module):
    """Bipartite matching loss between predicted and GT detections.

    The cost matrix for each sample is:

    .. math::

        C_{ij} = \\alpha_{\\text{cls}} \\, \\mathbb{1}[\\hat c_i \\neq c_j]
               + \\alpha_{\\text{pos}} \\, \\|\\hat p_i - p_j\\|_1
               + \\alpha_{\\text{emb}} \\, (1 - \\cos(\\hat e_i, e_j))

    The matching is solved with ``scipy.optimize.linear_sum_assignment``.
    The loss is the mean of matched costs.

    Parameters
    ----------
    alpha_cls : float
        Class-mismatch cost coefficient.
    alpha_pos : float
        L1 position cost coefficient.
    alpha_emb : float
        Cosine embedding distance cost coefficient.
    """

    def __init__(
        self,
        alpha_cls: float = 1.0,
        alpha_pos: float = 1.0,
        alpha_emb: float = 1.0,
    ) -> None:
        super().__init__()
        if linear_sum_assignment is None:
            raise ImportError(
                "HungarianMatchingLoss requires scipy.  "
                "Install it with: pip install scipy"
            )
        self.alpha_cls = alpha_cls
        self.alpha_pos = alpha_pos
        self.alpha_emb = alpha_emb

    def _build_cost_matrix(
        self,
        pred_classes: torch.Tensor,
        gt_classes: torch.Tensor,
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        pred_embeddings: torch.Tensor,
        gt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Build the (Np, Ng) cost matrix for a single sample.

        Parameters
        ----------
        pred_classes : LongTensor ``(Np,)``
        gt_classes : LongTensor ``(Ng,)``
        pred_positions : Tensor ``(Np, 3)``
        gt_positions : Tensor ``(Ng, 3)``
        pred_embeddings : Tensor ``(Np, D)``
        gt_embeddings : Tensor ``(Ng, D)``

        Returns
        -------
        Tensor ``(Np, Ng)``
        """
        Np = pred_classes.size(0)
        Ng = gt_classes.size(0)

        # Class mismatch: 1{pred_i != gt_j}
        cls_cost = (pred_classes.unsqueeze(1) != gt_classes.unsqueeze(0)).float()  # (Np, Ng)

        # L1 position distance
        pos_cost = torch.cdist(pred_positions.float(), gt_positions.float(), p=1)  # (Np, Ng)

        # Cosine distance: 1 - cos_sim
        pred_norm = F.normalize(pred_embeddings, p=2, dim=-1, eps=1e-8)
        gt_norm = F.normalize(gt_embeddings, p=2, dim=-1, eps=1e-8)
        cos_sim = torch.mm(pred_norm, gt_norm.t())  # (Np, Ng)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        emb_cost = 1.0 - cos_sim

        cost = (
            self.alpha_cls * cls_cost
            + self.alpha_pos * pos_cost
            + self.alpha_emb * emb_cost
        )
        cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
        return cost

    def forward(
        self,
        pred_classes: torch.Tensor,
        gt_classes: torch.Tensor,
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        pred_embeddings: torch.Tensor,
        gt_embeddings: torch.Tensor,
        pred_mask: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_classes : LongTensor ``(B, Np)``
        gt_classes : LongTensor ``(B, Ng)``
        pred_positions : Tensor ``(B, Np, 3)``
        gt_positions : Tensor ``(B, Ng, 3)``
        pred_embeddings : Tensor ``(B, Np, D)``
        gt_embeddings : Tensor ``(B, Ng, D)``
        pred_mask : BoolTensor ``(B, Np)``, optional
            ``True`` for valid predictions.
        gt_mask : BoolTensor ``(B, Ng)``, optional
            ``True`` for valid ground-truth entries.

        Returns
        -------
        Scalar matching loss (mean of matched costs across the batch).
        """
        B = pred_classes.size(0)
        device = pred_classes.device
        total_cost = torch.tensor(0.0, device=device)
        total_matches = 0

        for b in range(B):
            # Determine valid indices
            if pred_mask is not None:
                p_idx = pred_mask[b].nonzero(as_tuple=True)[0]
            else:
                p_idx = torch.arange(pred_classes.size(1), device=device)

            if gt_mask is not None:
                g_idx = gt_mask[b].nonzero(as_tuple=True)[0]
            else:
                g_idx = torch.arange(gt_classes.size(1), device=device)

            if len(p_idx) == 0 or len(g_idx) == 0:
                continue

            cost = self._build_cost_matrix(
                pred_classes[b, p_idx],
                gt_classes[b, g_idx],
                pred_positions[b, p_idx],
                gt_positions[b, g_idx],
                pred_embeddings[b, p_idx],
                gt_embeddings[b, g_idx],
            )

            # Solve assignment (on CPU via scipy)
            cost_np = cost.detach().cpu().numpy()
            cost_np = np.nan_to_num(cost_np, nan=1e4, posinf=1e4, neginf=-1e4)
            row_ind, col_ind = linear_sum_assignment(cost_np)

            matched_cost = cost[row_ind, col_ind].sum()
            total_cost = total_cost + matched_cost
            total_matches += len(row_ind)

        if total_matches == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_cost / total_matches


# ---------------------------------------------------------------------------
# Total Loss
# ---------------------------------------------------------------------------

class SAFEGroundTotalLoss(nn.Module):
    """Aggregate all SAFE-Ground sub-losses.

    .. math::

        \\mathcal{L} = \\mathcal{L}_{\\text{ver}}
                     + \\lambda_{\\text{om}} \\, \\mathcal{L}_{\\text{om}}
                     + \\lambda_{\\text{match}} \\, \\mathcal{L}_{\\text{match}}
                     + \\lambda_{\\text{LM}} \\, \\mathcal{L}_{\\text{LM}}

    Parameters
    ----------
    lambda_om : float
        Weight for omission loss.
    lambda_match : float
        Weight for Hungarian matching loss.
    lambda_lm : float
        Weight for language-modelling loss (passed through from the VLM).
    label_smoothing : float
        Label smoothing for the verification CE loss.
    alpha_cls : float
        Class-mismatch cost for Hungarian matching.
    alpha_pos : float
        L1 position cost for Hungarian matching.
    alpha_emb : float
        Cosine embedding cost for Hungarian matching.
    """

    def __init__(
        self,
        lambda_om: float = 1.0,
        lambda_match: float = 0.5,
        lambda_lm: float = 0.1,
        label_smoothing: float = 0.0,
        alpha_cls: float = 1.0,
        alpha_pos: float = 1.0,
        alpha_emb: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_om = lambda_om
        self.lambda_match = lambda_match
        self.lambda_lm = lambda_lm

        self.verification_loss = SeverityWeightedVerificationLoss(
            label_smoothing=label_smoothing,
        )
        self.omission_loss = SeverityWeightedOmissionLoss()
        self.matching_loss = HungarianMatchingLoss(
            alpha_cls=alpha_cls,
            alpha_pos=alpha_pos,
            alpha_emb=alpha_emb,
        )
        self.severity_computer = SeverityWeightComputer()

    def forward(
        self,
        # Head A inputs
        claim_logits: torch.Tensor,
        claim_targets: torch.Tensor,
        claim_mask: Optional[torch.Tensor],
        # Head B inputs
        omission_logits: torch.Tensor,
        omission_targets: torch.Tensor,
        evidence_mask: Optional[torch.Tensor],
        # Severity inputs (shared)
        class_ids: torch.Tensor,
        distances: torch.Tensor,
        path_ids: torch.Tensor,
        v_rel_norms: torch.Tensor,
        # Hungarian matching inputs
        pred_classes: torch.Tensor,
        gt_classes: torch.Tensor,
        pred_positions: torch.Tensor,
        gt_positions: torch.Tensor,
        pred_embeddings: torch.Tensor,
        gt_embeddings: torch.Tensor,
        pred_det_mask: Optional[torch.Tensor] = None,
        gt_det_mask: Optional[torch.Tensor] = None,
        # LM loss (pass-through scalar)
        lm_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            ``"total"``        — scalar total loss.
            ``"verification"`` — scalar verification loss.
            ``"omission"``     — scalar omission loss.
            ``"matching"``     — scalar Hungarian matching loss.
            ``"lm"``           — scalar LM loss (0 if not provided).
        """
        # Severity weights
        severity_weights = self.severity_computer(
            class_ids, distances, path_ids, v_rel_norms
        )

        # Adapt severity weights to claim / evidence dimensions.
        # In general the severity-weight tensor has shape (B, K) where K is
        # the number of objects.  For the claim loss we broadcast/truncate
        # to (B, N); for the omission loss to (B, M).  Here we handle size
        # differences gracefully by padding/truncating along dim=1.
        def _adapt_weights(w: torch.Tensor, target_len: int) -> torch.Tensor:
            """Pad or truncate ``w`` along dim-1 to ``target_len``."""
            cur = w.size(1)
            if cur >= target_len:
                return w[:, :target_len]
            pad = torch.ones(w.size(0), target_len - cur, device=w.device)
            return torch.cat([w, pad], dim=1)

        B_claim = claim_logits.size(1)
        B_evidence = omission_logits.squeeze(-1).size(1)

        claim_weights = _adapt_weights(severity_weights, B_claim)
        evidence_weights = _adapt_weights(severity_weights, B_evidence)

        # Sub-losses
        l_ver = self.verification_loss(
            claim_logits, claim_targets, claim_weights, claim_mask,
        )
        l_om = self.omission_loss(
            omission_logits, omission_targets, evidence_weights, evidence_mask,
        )
        l_match = self.matching_loss(
            pred_classes, gt_classes,
            pred_positions, gt_positions,
            pred_embeddings, gt_embeddings,
            pred_det_mask, gt_det_mask,
        )
        l_lm = lm_loss if lm_loss is not None else torch.tensor(0.0, device=claim_logits.device)

        total = l_ver + self.lambda_om * l_om + self.lambda_match * l_match + self.lambda_lm * l_lm

        return {
            "total": total,
            "verification": l_ver,
            "omission": l_om,
            "matching": l_match,
            "lm": l_lm,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, N, M, D, K = 2, 6, 10, 128, 10

    print("=" * 60)
    print("Smoke test: SeverityWeightComputer")
    print("=" * 60)
    swc = SeverityWeightComputer()
    class_ids = torch.randint(0, 6, (B, K))
    distances = torch.rand(B, K) * 80
    path_ids = torch.randint(0, 3, (B, K))
    v_rel = torch.rand(B, K) * 20
    weights = swc(class_ids, distances, path_ids, v_rel)
    print(f"  weights shape: {weights.shape}")
    print(f"  sample weights: {weights[0, :5].tolist()}")
    assert weights.shape == (B, K)
    print("  ✓ SeverityWeightComputer passed.\n")

    print("=" * 60)
    print("Smoke test: SeverityWeightedVerificationLoss")
    print("=" * 60)
    ver_loss_fn = SeverityWeightedVerificationLoss()
    logits = torch.randn(B, N, 3, device=device, requires_grad=True)
    targets = torch.randint(0, 3, (B, N), device=device)
    w = torch.ones(B, N, device=device) * 2.0
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[0, -2:] = False
    l_ver = ver_loss_fn(logits, targets, w, mask)
    print(f"  verification loss: {l_ver.item():.4f}")
    l_ver.backward()
    print("  ✓ backward passed.\n")

    print("=" * 60)
    print("Smoke test: SeverityWeightedOmissionLoss")
    print("=" * 60)
    om_loss_fn = SeverityWeightedOmissionLoss()
    om_logits = torch.randn(B, M, 1, device=device, requires_grad=True)
    om_targets = torch.randint(0, 2, (B, M), device=device).float()
    om_w = torch.ones(B, M, device=device) * 1.5
    om_mask = torch.ones(B, M, dtype=torch.bool, device=device)
    om_mask[1, -3:] = False
    l_om = om_loss_fn(om_logits, om_targets, om_w, om_mask)
    print(f"  omission loss: {l_om.item():.4f}")
    l_om.backward()
    print("  ✓ backward passed.\n")

    print("=" * 60)
    print("Smoke test: HungarianMatchingLoss")
    print("=" * 60)
    match_loss_fn = HungarianMatchingLoss()
    Np, Ng = 8, 6
    pred_cls = torch.randint(0, 5, (B, Np), device=device)
    gt_cls = torch.randint(0, 5, (B, Ng), device=device)
    pred_pos = torch.randn(B, Np, 3, device=device)
    gt_pos = torch.randn(B, Ng, 3, device=device)
    pred_emb = torch.randn(B, Np, D, device=device, requires_grad=True)
    gt_emb = torch.randn(B, Ng, D, device=device)
    l_match = match_loss_fn(pred_cls, gt_cls, pred_pos, gt_pos, pred_emb, gt_emb)
    print(f"  matching loss: {l_match.item():.4f}")
    l_match.backward()
    print(f"  pred_emb grad norm: {pred_emb.grad.norm().item():.6f}")
    print("  ✓ backward passed.\n")

    print("=" * 60)
    print("Smoke test: SAFEGroundTotalLoss")
    print("=" * 60)
    total_loss_fn = SAFEGroundTotalLoss(
        lambda_om=1.0,
        lambda_match=0.5,
        lambda_lm=0.1,
    )

    claim_logits = torch.randn(B, N, 3, device=device, requires_grad=True)
    claim_targets = torch.randint(0, 3, (B, N), device=device)
    claim_mask = torch.ones(B, N, dtype=torch.bool, device=device)

    om_logits = torch.randn(B, M, 1, device=device, requires_grad=True)
    om_targets = torch.randint(0, 2, (B, M), device=device).float()
    ev_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    sev_cls = torch.randint(0, 6, (B, K), device=device)
    sev_dist = torch.rand(B, K, device=device) * 60
    sev_path = torch.randint(0, 3, (B, K), device=device)
    sev_vrel = torch.rand(B, K, device=device) * 15

    pred_emb2 = torch.randn(B, Np, D, device=device, requires_grad=True)
    gt_emb2 = torch.randn(B, Ng, D, device=device)
    fake_lm = torch.tensor(2.5, device=device, requires_grad=True)

    losses = total_loss_fn(
        claim_logits=claim_logits,
        claim_targets=claim_targets,
        claim_mask=claim_mask,
        omission_logits=om_logits,
        omission_targets=om_targets,
        evidence_mask=ev_mask,
        class_ids=sev_cls,
        distances=sev_dist,
        path_ids=sev_path,
        v_rel_norms=sev_vrel,
        pred_classes=pred_cls,
        gt_classes=gt_cls,
        pred_positions=pred_pos,
        gt_positions=gt_pos,
        pred_embeddings=pred_emb2,
        gt_embeddings=gt_emb2,
        lm_loss=fake_lm,
    )

    print(f"  total:        {losses['total'].item():.4f}")
    print(f"  verification: {losses['verification'].item():.4f}")
    print(f"  omission:     {losses['omission'].item():.4f}")
    print(f"  matching:     {losses['matching'].item():.4f}")
    print(f"  lm:           {losses['lm'].item():.4f}")

    losses["total"].backward()
    print(f"  claim_logits grad norm: {claim_logits.grad.norm().item():.6f}")
    print("  ✓ backward passed.\n")

    print("All loss smoke tests passed! ✅")

"""
SAFE-Ground — Evidence-Cross-Attention Verifier, Sensor Evidence Encoder,
and Claim Embedding Extractor.

Modules
-------
SensorEvidenceEncoder
    Encodes raw 3-D object detections (class id + continuous features)
    into dense evidence tokens of dimension *D*.

ClaimEmbeddingExtractor
    Gathers per-claim embeddings from VLM hidden states using
    [CLAIM_START] span indices.

EvidenceCrossAttentionVerifier
    Confidence-gated cross-attention between claim embeddings and
    evidence tokens, with two task heads:
        Head A — 3-class claim verification (supported / contradicted / unverifiable)
        Head B — binary omission detection per evidence token
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sensor Evidence Encoder
# ---------------------------------------------------------------------------

class SensorEvidenceEncoder(nn.Module):
    """Encode raw detection features into dense evidence tokens.

    Each detection is an 18-dim vector::

        [class_id (int),
         x, y, z, l, w, h, vx, vy, vz, confidence,
         cam_visible_0 … cam_visible_5]          # 6 binary flags

    Processing pipeline:
        1. ``class_id`` → learned class embedding (``class_embed_dim``).
        2. Remaining 17 continuous features concatenated with class embedding.
        3. Two-layer MLP → output of dimension *D*.
        4. A learned frame-position embedding is added when detections from
           multiple temporal frames are concatenated.

    Parameters
    ----------
    num_classes : int
        Number of object classes (for the embedding table).
    hidden_dim : int
        Output token dimension *D*.
    class_embed_dim : int
        Dimensionality of the learned class embedding.
    num_continuous : int
        Number of continuous input features (default 17).
    max_frames : int
        Maximum number of temporal frames (for frame-position embedding).
    mlp_dropout : float
        Dropout probability inside the MLP.
    """

    def __init__(
        self,
        num_classes: int = 24,
        hidden_dim: int = 3584,
        class_embed_dim: int = 16,
        num_continuous: int = 17,
        max_frames: int = 8,
        mlp_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.class_embed_dim = class_embed_dim
        self.num_continuous = num_continuous

        # 1. Class embedding
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        # 2. Projection MLP: (class_embed_dim + num_continuous) → hidden_dim
        mlp_input_dim = class_embed_dim + num_continuous
        self.projection = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(mlp_dropout),
        )

        # 3. Temporal (frame-position) embedding
        self.frame_position_embedding = nn.Embedding(max_frames, hidden_dim)

        # Layer norm after projection
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        class_ids: torch.Tensor,
        continuous_features: torch.Tensor,
        frame_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        class_ids : Tensor, shape ``(B, M)``
            Integer class IDs per detection.
        continuous_features : Tensor, shape ``(B, M, num_continuous)``
            The 17 continuous detection features.
        frame_ids : Tensor, shape ``(B, M)``, optional
            Frame index per detection (0-indexed).  If *None*, all
            detections are assumed to belong to frame 0.

        Returns
        -------
        Tensor, shape ``(B, M, D)``
            Dense evidence tokens.
        """
        B, M = class_ids.shape

        # Class embedding
        cls_emb = self.class_embedding(class_ids)  # (B, M, class_embed_dim)

        # Concatenate
        x = torch.cat([cls_emb, continuous_features], dim=-1)  # (B, M, class_embed_dim + num_continuous)

        # MLP projection
        x = self.projection(x)  # (B, M, D)

        # Add frame-position embedding
        if frame_ids is None:
            frame_ids = torch.zeros(B, M, dtype=torch.long, device=x.device)
        x = x + self.frame_position_embedding(frame_ids)

        x = self.layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Claim Embedding Extractor
# ---------------------------------------------------------------------------

class ClaimEmbeddingExtractor(nn.Module):
    """Extract per-claim embeddings from VLM hidden states.

    Given the full sequence of hidden states from the VLM and a list of
    ``(start, end)`` span indices that delimit each claim, this module
    extracts the hidden state at the ``[CLAIM_START]`` position (i.e.
    ``start_i``) for every claim.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the VLM hidden states (*D*).
    max_claims : int
        Maximum number of claims *N* per sample (for padding).
    """

    def __init__(self, hidden_dim: int = 3584, max_claims: int = 32) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_claims = max_claims

    def forward(
        self,
        hidden_states: torch.Tensor,
        claim_spans: List[List[Tuple[int, int]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : Tensor, shape ``(B, SeqLen, D)``
            VLM hidden states for the full sequence.
        claim_spans : list of list of (start, end) tuples
            ``claim_spans[b]`` contains the span indices for sample *b*.
            Each span ``(start_i, end_i)`` marks a claim; we extract the
            hidden state at index ``start_i`` (the ``[CLAIM_START]`` token).

        Returns
        -------
        claim_embeddings : Tensor, shape ``(B, N, D)``
            Extracted claim embeddings, zero-padded to ``max_claims``.
        claim_mask : Tensor, shape ``(B, N)``
            Boolean mask — ``True`` for real claims, ``False`` for padding.
        """
        B, _SeqLen, D = hidden_states.shape
        N = self.max_claims
        device = hidden_states.device

        claim_embeddings = torch.zeros(B, N, D, device=device, dtype=hidden_states.dtype)
        claim_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            spans = claim_spans[b]
            num_claims = min(len(spans), N)
            for i in range(num_claims):
                start_idx = spans[i][0]
                claim_embeddings[b, i] = hidden_states[b, start_idx]
                claim_mask[b, i] = True

        return claim_embeddings, claim_mask


# ---------------------------------------------------------------------------
# Evidence-Cross-Attention Verifier
# ---------------------------------------------------------------------------

class ConfidenceGatedCrossAttention(nn.Module):
    """Multi-head cross-attention with confidence gating.

    Queries come from *claim* embeddings; keys and values come from
    *evidence* tokens.  Before the softmax, ``log(sigma_j + eps)`` is
    added to the raw attention logits, where ``sigma_j`` is the detection
    confidence of evidence token *j*.  This down-weights uncertain
    detections in a numerically stable, differentiable manner.

    Parameters
    ----------
    hidden_dim : int
        Model dimension *D*.
    num_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    eps : float
        Epsilon for the log-confidence bias.
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 8,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        queries : Tensor ``(B, Nq, D)``
        keys : Tensor ``(B, Nk, D)``
        values : Tensor ``(B, Nk, D)``
        confidence : Tensor ``(B, Nk)``, optional
            Detection confidence scores in [0, 1].  If given, the log-
            confidence bias is added to the attention logits.
        key_padding_mask : Tensor ``(B, Nk)``, optional
            Boolean mask — ``True`` for positions to **ignore** (padding).

        Returns
        -------
        Tensor ``(B, Nq, D)``
        """
        B, Nq, _ = queries.shape
        _, Nk, _ = keys.shape

        # Project
        Q = self.q_proj(queries).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        K = self.k_proj(keys).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, Nk, Hd)
        V = self.v_proj(values).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)      # (B, H, Nk, Hd)

        # Scaled dot-product attention logits
        scale = math.sqrt(self.head_dim)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, Nq, Nk)

        # Confidence gating: add log(sigma_j + eps) as bias
        if confidence is not None:
            conf_bias = torch.log(confidence + self.eps)  # (B, Nk)
            conf_bias = conf_bias.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nk)
            attn_logits = attn_logits + conf_bias

        # Key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: True = pad → mask out with -1e9
            # If all keys in a sample are padding, avoid all -inf which causes softmax to produce NaN
            all_padding = key_padding_mask.all(dim=-1, keepdim=True)  # (B, 1)
            safe_pad_mask = key_padding_mask & ~all_padding
            attn_logits = attn_logits.masked_fill(
                safe_pad_mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, Nk)
                -1e9,
            )

        attn_weights = F.softmax(attn_logits, dim=-1)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(all_padding.unsqueeze(1).unsqueeze(2), 0.0)

        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, Nq, Hd)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.hidden_dim)
        out = self.out_proj(out)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out


class EvidenceCrossAttentionVerifier(nn.Module):
    """Full verifier module with two task heads.

    Architecture
    ------------
    1. **Forward cross-attention** — claims attend over evidence tokens
       (confidence-gated), followed by residual + LayerNorm + FFN +
       residual + LayerNorm.
    2. **Head A (Claim Verifier)** — ``Linear(D, 3)`` on claim
       representations → per-claim logits for
       ``{supported, contradicted, unverifiable}``.
    3. **Reverse cross-attention** — evidence tokens attend over claim
       embeddings (standard, no confidence gating), followed by residual
       + LayerNorm + FFN + residual + LayerNorm.
    4. **Head B (Omission Head)** — ``Linear(D, 1)`` on evidence
       representations → per-evidence logit for ``{mentioned=0, omitted=1}``.

    Parameters
    ----------
    hidden_dim : int
        Model dimension *D*.
    num_heads : int
        Number of attention heads in cross-attention layers.
    ffn_dim : int
        Inner dimension of the position-wise feed-forward networks.
        Defaults to ``4 * hidden_dim``.
    dropout : float
        Dropout probability.
    num_verification_classes : int
        Number of verification labels (default 3).
    confidence_eps : float
        Epsilon for confidence gating.
    """

    def __init__(
        self,
        hidden_dim: int = 3584,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_verification_classes: int = 3,
        confidence_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if ffn_dim is None:
            ffn_dim = 4 * hidden_dim

        # --- Forward cross-attention (claims → evidence) -----------------
        self.fwd_cross_attn = ConfidenceGatedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            eps=confidence_eps,
        )
        self.fwd_norm1 = nn.LayerNorm(hidden_dim)
        self.fwd_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.fwd_norm2 = nn.LayerNorm(hidden_dim)

        # --- Head A: claim verification ----------------------------------
        self.head_a = nn.Linear(hidden_dim, num_verification_classes)

        # --- Reverse cross-attention (evidence → claims) -----------------
        self.rev_cross_attn = ConfidenceGatedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            eps=confidence_eps,
        )
        self.rev_norm1 = nn.LayerNorm(hidden_dim)
        self.rev_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.rev_norm2 = nn.LayerNorm(hidden_dim)

        # --- Head B: omission detection ----------------------------------
        self.head_b = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        claim_embeddings: torch.Tensor,
        evidence_tokens: torch.Tensor,
        evidence_confidence: torch.Tensor,
        claim_mask: Optional[torch.Tensor] = None,
        evidence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        claim_embeddings : Tensor ``(B, N, D)``
            Per-claim embeddings.
        evidence_tokens : Tensor ``(B, M, D)``
            Dense evidence tokens from the sensor encoder.
        evidence_confidence : Tensor ``(B, M)``
            Detection confidence scores in [0, 1].
        claim_mask : Tensor ``(B, N)``, optional
            Boolean — ``True`` for real claims, ``False`` for padding.
        evidence_mask : Tensor ``(B, M)``, optional
            Boolean — ``True`` for real evidence, ``False`` for padding.

        Returns
        -------
        dict with keys:
            ``"claim_logits"``   — Tensor ``(B, N, 3)`` — Head A logits.
            ``"omission_logits"`` — Tensor ``(B, M, 1)`` — Head B logits.
        """
        # Convert valid-element masks → padding masks (True = ignore)
        evidence_pad_mask = ~evidence_mask if evidence_mask is not None else None
        claim_pad_mask = ~claim_mask if claim_mask is not None else None

        # === Forward cross-attention: claims attend over evidence ========
        fwd_attn_out = self.fwd_cross_attn(
            queries=claim_embeddings,
            keys=evidence_tokens,
            values=evidence_tokens,
            confidence=evidence_confidence,
            key_padding_mask=evidence_pad_mask,
        )
        claim_repr = self.fwd_norm1(claim_embeddings + fwd_attn_out)
        claim_repr = self.fwd_norm2(claim_repr + self.fwd_ffn(claim_repr))

        # Head A
        claim_logits = self.head_a(claim_repr)  # (B, N, 3)

        # === Reverse cross-attention: evidence attends over claims =======
        rev_attn_out = self.rev_cross_attn(
            queries=evidence_tokens,
            keys=claim_repr,
            values=claim_repr,
            confidence=None,  # no confidence gating in reverse direction
            key_padding_mask=claim_pad_mask,
        )
        evidence_repr = self.rev_norm1(evidence_tokens + rev_attn_out)
        evidence_repr = self.rev_norm2(evidence_repr + self.rev_ffn(evidence_repr))

        # Head B
        omission_logits = self.head_b(evidence_repr)  # (B, M, 1)

        return {
            "claim_logits": claim_logits,
            "omission_logits": omission_logits,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, N, M, D = 2, 8, 12, 256  # small dims for quick testing
    num_classes = 10

    print("=" * 60)
    print("Smoke test: SensorEvidenceEncoder")
    print("=" * 60)

    encoder = SensorEvidenceEncoder(
        num_classes=num_classes,
        hidden_dim=D,
        class_embed_dim=16,
        num_continuous=17,
        max_frames=4,
    ).to(device)

    class_ids = torch.randint(0, num_classes, (B, M), device=device)
    cont_feats = torch.randn(B, M, 17, device=device)
    frame_ids = torch.randint(0, 4, (B, M), device=device)

    evidence_tokens = encoder(class_ids, cont_feats, frame_ids)
    print(f"  Input class_ids:          {class_ids.shape}")
    print(f"  Input continuous_features:{cont_feats.shape}")
    print(f"  Output evidence_tokens:   {evidence_tokens.shape}")
    assert evidence_tokens.shape == (B, M, D), f"Expected {(B, M, D)}, got {evidence_tokens.shape}"
    print("  ✓ SensorEvidenceEncoder passed.\n")

    print("=" * 60)
    print("Smoke test: ClaimEmbeddingExtractor")
    print("=" * 60)

    extractor = ClaimEmbeddingExtractor(hidden_dim=D, max_claims=N).to(device)
    seq_len = 128
    hidden_states = torch.randn(B, seq_len, D, device=device)
    claim_spans = [
        [(5, 10), (20, 25), (40, 45)],          # 3 claims for sample 0
        [(2, 8), (15, 20), (30, 35), (50, 55)],  # 4 claims for sample 1
    ]
    claim_embs, claim_mask = extractor(hidden_states, claim_spans)
    print(f"  Input hidden_states: {hidden_states.shape}")
    print(f"  Output claim_embs:   {claim_embs.shape}")
    print(f"  Output claim_mask:   {claim_mask.shape}")
    print(f"  Claim counts:        {claim_mask.sum(dim=1).tolist()}")
    assert claim_embs.shape == (B, N, D)
    assert claim_mask.shape == (B, N)
    print("  ✓ ClaimEmbeddingExtractor passed.\n")

    print("=" * 60)
    print("Smoke test: EvidenceCrossAttentionVerifier")
    print("=" * 60)

    verifier = EvidenceCrossAttentionVerifier(
        hidden_dim=D,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
    ).to(device)

    evidence_conf = torch.rand(B, M, device=device)
    evidence_mask = torch.ones(B, M, dtype=torch.bool, device=device)
    evidence_mask[0, -3:] = False  # pad last 3 for sample 0

    outputs = verifier(
        claim_embeddings=claim_embs,
        evidence_tokens=evidence_tokens,
        evidence_confidence=evidence_conf,
        claim_mask=claim_mask,
        evidence_mask=evidence_mask,
    )

    print(f"  claim_logits:   {outputs['claim_logits'].shape}")
    print(f"  omission_logits:{outputs['omission_logits'].shape}")
    assert outputs["claim_logits"].shape == (B, N, 3)
    assert outputs["omission_logits"].shape == (B, M, 1)
    print("  ✓ EvidenceCrossAttentionVerifier passed.\n")

    total_params = sum(p.numel() for p in verifier.parameters())
    print(f"Verifier total parameters: {total_params:,}")
    print("All smoke tests passed! ✅")

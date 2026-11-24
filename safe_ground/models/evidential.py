"""Pillar 2: Evidential Cross-Attention & Subjective Logic Head.

Parameterizes claim verification as a Dirichlet prior:
  alpha = Softplus(logits) + 1.0
  S = sum(alpha)
  Belief (b_k) = (alpha_k - 1) / S
  Vacuity (u) = K / S

Where Vacuity represents epistemic ignorance (e.g. sensor occlusion / blindspot).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EvidentialCrossAttentionHead(nn.Module):
    """Outputs Dirichlet evidence parameters for 3-class claim verification."""

    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, claim_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Maps hidden representations to Dirichlet concentration parameters and Subjective Logic tuples.
        
        claim_repr: (B, N, D)
        """
        raw_logits = self.proj(claim_repr)  # (B, N, K)
        # alpha_k >= 1.0 for valid Dirichlet distribution
        evidence = F.softplus(raw_logits)
        alpha = evidence + 1.0  # (B, N, K)
        total_strength = torch.sum(alpha, dim=-1, keepdim=True)  # S = sum(alpha_k)

        # Beliefs b_k and Epistemic Vacuity u
        beliefs = (alpha - 1.0) / total_strength  # (B, N, K)
        vacuity = self.num_classes / total_strength  # (B, N, 1)

        # Expected probability under Dirichlet
        expected_prob = alpha / total_strength

        return {
            "alpha": alpha,
            "total_strength": total_strength,
            "beliefs": beliefs,
            "vacuity": vacuity,
            "expected_prob": expected_prob,
            "raw_logits": raw_logits,
        }


class EvidentialVerificationLoss(nn.Module):
    """Computes Evidential Loss with Digamma function and KL divergence regularization.
    
    Attenuates verification penalties when sensor epistemic vacuity is high.
    """

    def __init__(self, kl_weight: float = 0.05):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(
        self,
        alpha: torch.Tensor,
        targets: torch.Tensor,
        severity_weights: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """alpha: (B, N, K)
        targets: (B, N) with values in {0, 1, 2}
        severity_weights: (B, N)
        mask: (B, N) bool
        """
        B, N, K = alpha.shape
        S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, N, 1)

        # One-hot encode targets
        one_hot = F.one_hot(targets.clamp(0, K - 1), num_classes=K).float()  # (B, N, K)

        # 1. Expected Cross-Entropy under Dirichlet: sum(y_k * (psi(S) - psi(alpha_k)))
        psi_S = torch.digamma(S)
        psi_alpha = torch.digamma(alpha)
        loss_ce = torch.sum(one_hot * (psi_S - psi_alpha), dim=-1)  # (B, N)

        # 2. KL Divergence Regularization (pushes uninformative evidence to uniform Dir(1,1,1))
        alpha_tilde = one_hot + (1.0 - one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)

        first_term = (
            torch.lgamma(S_tilde.squeeze(-1))
            - torch.sum(torch.lgamma(alpha_tilde), dim=-1)
            + torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=-1)
            - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        )
        second_term = torch.sum(
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)),
            dim=-1,
        )
        kl_div = first_term + second_term

        # Combined evidential loss, weighted by severity
        sample_loss = (loss_ce + self.kl_weight * kl_div) * severity_weights

        # Normalize by valid mask
        valid_loss = sample_loss[mask]
        norm = severity_weights[mask].sum() + 1e-6
        return valid_loss.sum() / norm


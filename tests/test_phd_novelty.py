"""Unit tests for PhD Novelty Pillars (Evidential Cross-Attention, Reachability FRS, and CBF Shield)."""

import unittest
import torch
from safe_ground.models.evidential import EvidentialCrossAttentionHead, EvidentialVerificationLoss
from safe_ground.utils.reachability import KinematicReachabilityComputer
from safe_ground.mitigation.cbf_shield import NeuroSymbolicSafetyShield


class TestPhDNoveltyComponents(unittest.TestCase):

    def test_evidential_head_properties(self):
        B, N, D = 2, 4, 128
        head = EvidentialCrossAttentionHead(hidden_dim=D, num_classes=3)
        dummy_claims = torch.randn(B, N, D)
        out = head(dummy_claims)

        # 1. Check alpha >= 1.0 (valid Dirichlet)
        self.assertTrue(torch.all(out["alpha"] >= 1.0))

        # 2. Check Subjective Logic constraint: sum(beliefs) + vacuity == 1.0
        beliefs_sum = torch.sum(out["beliefs"], dim=-1, keepdim=True)
        total_evidence = beliefs_sum + out["vacuity"]
        ones = torch.ones_like(total_evidence)
        self.assertTrue(torch.allclose(total_evidence, ones, atol=1e-5))

        # 3. Test Evidential Loss and backward
        loss_fn = EvidentialVerificationLoss()
        targets = torch.randint(0, 3, (B, N))
        weights = torch.ones(B, N) * 2.0
        mask = torch.ones(B, N, dtype=torch.bool)
        loss = loss_fn(out["alpha"], targets, weights, mask)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

    def test_kinematic_reachability_frs(self):
        reach_comp = KinematicReachabilityComputer()

        # In-path approaching object should have much higher FRS integral than lateral stationary object
        # Approaching car directly ahead (x=15m, y=0m, vx=-5m/s toward ego)
        frs_approaching = reach_comp.compute_frs_collision_integral(
            x_obj=15.0, y_obj=0.0, vx_obj=-5.0, vy_obj=0.0
        )
        # Stationary object far to the side (x=15m, y=10m, vx=0, vy=0)
        frs_sidewalk = reach_comp.compute_frs_collision_integral(
            x_obj=15.0, y_obj=10.0, vx_obj=0.0, vy_obj=0.0
        )

        self.assertGreater(frs_approaching, frs_sidewalk)

        # FRS severity weight for approaching VRU
        w_vru_collision = reach_comp.compute_frs_severity_weight("pedestrian", x=12.0, y=0.5, vx=-1.0)
        w_vru_distant = reach_comp.compute_frs_severity_weight("pedestrian", x=40.0, y=8.0)
        self.assertGreater(w_vru_collision, w_vru_distant)

    def test_cbf_safety_shield(self):
        shield = NeuroSymbolicSafetyShield(nominal_speed_mps=10.0)

        # Safe scenario: low risk, low vacuity
        ctrl_safe = shield.compute_safe_control(
            predicted_hallucination_risk=0.5, epistemic_vacuity=0.05, nearest_obstacle_dist=30.0
        )
        self.assertEqual(ctrl_safe["action"], "NORMAL")
        self.assertAlmostEqual(ctrl_safe["commanded_speed_mps"], 10.0, delta=0.5)

        # Critical scenario: high risk hallucination in ego path at close range
        ctrl_emergency = shield.compute_safe_control(
            predicted_hallucination_risk=28.0, epistemic_vacuity=0.4, nearest_obstacle_dist=8.0
        )
        self.assertEqual(ctrl_emergency["action"], "EMERGENCY_BRAKE")
        self.assertEqual(ctrl_emergency["commanded_speed_mps"], 0.0)


if __name__ == "__main__":
    unittest.main()


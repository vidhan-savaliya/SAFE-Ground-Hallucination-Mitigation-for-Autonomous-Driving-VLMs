"""Unit tests for HallucinationInjector."""

import unittest
from safe_ground.data.schema import Claim, Detection3D
from safe_ground.data.hallucination_injector import HallucinationInjector


class TestHallucinationInjector(unittest.TestCase):

    def setUp(self):
        self.injector = HallucinationInjector(
            commission_prob=1.0,
            omission_prob=1.0,
            attribute_prob=1.0,
            seed=42,
        )

    def test_injection_produces_contradictions_and_omissions(self):
        clean_claims = [
            Claim(object_class="car", bearing_sector="front", distance_bin="medium"),
            Claim(object_class="truck", bearing_sector="left", distance_bin="far"),
        ]
        clean_evidence = [
            Detection3D(class_name="car", class_id=3, x=15.0, y=0.0, z=0.0, l=4.5, w=1.8, h=1.5),
            Detection3D(class_name="truck", class_id=4, x=20.0, y=5.0, z=0.0, l=8.0, w=2.5, h=3.0),
        ]

        corrupted_claims, corrupted_ev, meta = self.injector.process_sample(
            clean_claims, clean_evidence
        )

        # Ensure that injection took place
        self.assertTrue(meta["num_commissions"] > 0 or meta["num_omissions"] > 0 or meta["num_attribute_errors"] > 0)
        # Check claim statuses are set
        for c in corrupted_claims:
            self.assertIn(c.ground_truth_status, ["supported", "contradicted", "unverifiable"])


if __name__ == "__main__":
    unittest.main()


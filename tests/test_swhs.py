"""Unit tests for SWHS calculation."""

import unittest
from safe_ground.data.schema import Claim, Detection3D
from safe_ground.eval.swhs import SWHSCalculator, compute_object_severity


class TestSWHS(unittest.TestCase):

    def setUp(self):
        self.calc = SWHSCalculator()

    def test_vru_weighted_higher_than_vehicle(self):
        # A pedestrian at 5m vs car at 5m
        w_ped = compute_object_severity("pedestrian", x=5.0, y=0.0)
        w_car = compute_object_severity("car", x=5.0, y=0.0)
        self.assertGreater(w_ped, w_car)

    def test_distance_decay(self):
        # Near object (5m) should have higher severity than distant object (40m)
        w_near = compute_object_severity("pedestrian", x=5.0, y=0.0)
        w_far = compute_object_severity("pedestrian", x=40.0, y=0.0)
        self.assertGreater(w_near, w_far)

    def test_in_path_weighted_higher_than_off_path(self):
        # In-path (lateral y=0.5m) vs off-path (lateral y=8.0m)
        w_in = compute_object_severity("car", x=15.0, y=0.5)
        w_off = compute_object_severity("car", x=15.0, y=8.0)
        self.assertGreater(w_in, w_off)

    def test_clean_sample_zero_swhs(self):
        claims = [
            Claim(object_class="car", bearing_sector="front", distance_bin="near", ground_truth_status="supported")
        ]
        gt_objects = [
            {"class_name": "car", "x": 10.0, "y": 0.0, "z": 0.0, "is_omitted": False}
        ]
        res = self.calc.compute_sample_swhs(claims, gt_objects)
        self.assertEqual(res["swhs_com"], 0.0)
        self.assertEqual(res["swhs_om"], 0.0)
        self.assertEqual(res["swhs"], 0.0)


if __name__ == "__main__":
    unittest.main()


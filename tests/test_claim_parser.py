"""Unit tests for ClaimParser."""

import unittest
from safe_ground.data.claim_parser import ClaimParser


class TestClaimParser(unittest.TestCase):

    def setUp(self):
        self.parser = ClaimParser()

    def test_strict_json_block(self):
        text = """
        Here is the scene:
        ```json
        [
          {"object_class": "pedestrian", "bearing_sector": "front_left", "distance_bin": "near", "attribute": "crossing"}
        ]
        ```
        """
        claims = self.parser.parse(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].object_class, "pedestrian")
        self.assertEqual(claims[0].bearing_sector, "front_left")
        self.assertEqual(claims[0].distance_bin, "near")

    def test_malformed_json_trailing_comma(self):
        text = """[{"object_class": "car", "bearing_sector": "front", "distance_bin": "medium",},]"""
        claims = self.parser.parse(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].object_class, "car")

    def test_tag_format_extraction(self):
        text = """We see <claim class="truck" bearing="back" dist="far" attr="parked"/>."""
        claims = self.parser.parse(text)
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].object_class, "truck")
        self.assertEqual(claims[0].bearing_sector, "back")
        self.assertEqual(claims[0].distance_bin, "far")

    def test_heuristic_regex_fallback(self):
        text = "A cyclist is nearby on the front left."
        claims = self.parser.parse(text)
        self.assertGreaterEqual(len(claims), 1)
        self.assertEqual(claims[0].object_class, "cyclist")
        self.assertEqual(claims[0].bearing_sector, "front_left")
        self.assertEqual(claims[0].distance_bin, "near")


if __name__ == "__main__":
    unittest.main()


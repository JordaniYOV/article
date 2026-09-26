"""Offline contract checks; these are not scientific benchmark evaluations."""

from dataclasses import asdict, fields
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.contracts import GroundTruth, ModelRequest, ModelResponse


class ContractTests(unittest.TestCase):
    def test_model_request_has_no_target_fields(self):
        expected = {"request_id", "image_paths", "prompt", "allowed_answers",
                    "max_new_tokens", "do_sample"}
        self.assertEqual({field.name for field in fields(ModelRequest)}, expected)
        request = ModelRequest("r1", ("image.png",), "Rock?", ("YES", "NO"))
        self.assertFalse(asdict(request)["do_sample"])

    def test_empty_images_rejected(self):
        with self.assertRaises(ValueError):
            ModelRequest("r1", (), "Rock?", ("YES", "NO"))

    def test_nonpositive_token_limit_rejected(self):
        with self.assertRaises(ValueError):
            ModelRequest("r1", ("image.png",), "Rock?", ("YES", "NO"), 0)

    def test_evaluation_join_is_explicit(self):
        target = GroundTruth("r1", "YES", "sample1", "scene1", "rule-v1")
        response = ModelResponse("r1", "YES", "ok", 0.1)
        self.assertEqual(target.request_id, response.request_id)
        self.assertNotIn("answer", asdict(response))


if __name__ == "__main__":
    unittest.main()

"""Offline contract checks; these are not scientific benchmark evaluations."""

from dataclasses import asdict
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.contracts import GroundTruth, ModelRequest, ModelResponse
from planetary_vlm.datasets.manifest import request_from_dict


class ContractTests(unittest.TestCase):
    def test_model_request_has_no_target_fields(self):
        request = ModelRequest("r1", ("image.png",), "Rock?", ("YES", "NO"))
        self.assertFalse(asdict(request)["do_sample"])
        for target_field in ("answer", "ground_truth", "mask_path", "outcome_telemetry"):
            with self.subTest(field=target_field), self.assertRaises(ValueError):
                request_from_dict({**asdict(request), target_field: "secret"})

    def test_malformed_image_path_rejected(self):
        with self.assertRaises(ValueError):
            ModelRequest("r1", ("",), "Rock?", ("YES", "NO"))

    def test_text_and_native_requests_use_same_contract(self):
        native = ModelRequest("r1", (), "Describe tile", (), modality_paths=(("native_tile", "tile"),))
        text = ModelRequest("r2", (), "Describe lunar geology", ())
        self.assertEqual(native.modality_paths[0][0], "native_tile")
        self.assertEqual(text.image_paths, ())

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

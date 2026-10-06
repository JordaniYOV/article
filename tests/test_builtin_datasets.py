"""Local-first catalog and restoration using synthetic bytes, never network/GPU."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from PIL import Image
    from fastapi.testclient import TestClient
    from api.main import create_app
    from api.settings import Settings
    from planetary_vlm.datasets.orbital import download_dataset, inspect_subset
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Install API test dependencies")
class BuiltinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = BytesIO()
        Image.new("L", (16, 16), 100).save(image, format="PNG")
        self.pixels = image.getvalue()
        self.hash = hashlib.sha256(self.pixels).hexdigest()
        self.spec = {"tracks": {p: {t: [] for t in ("segmentation", "height")} for p in ("mars", "moon")}}
        for planet in self.spec["tracks"]:
            for track in self.spec["tracks"][planet]:
                for index, split in enumerate(("test", "val"), 1):
                    prefix = f"data_orbital/orbital_300_v3/{planet}/{track}/{index:03d}"
                    target = "mask" if track == "segmentation" else "height_map"
                    row = {"sample_id": f"{planet}_{track}_{index}", "planet": planet,
                        "source_split": split, "source_dataset": "fixture/source", "source_revision": "pinned",
                        "scene_group_id": f"scene-{index}", "evaluation_eligible": True,
                        "image": f"{prefix}/image.png", "image_sha256": self.hash,
                        target: f"{prefix}/target.png", target + "_sha256": self.hash,
                        "mask_semantics": "imp" if index == 1 and planet == "moon" else "crater"}
                    self.spec["tracks"][planet][track].append(row)
                    for key in ("image", target):
                        path = self.root / row[key]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(self.pixels)
                manifest = self.root / f"data_orbital/orbital_300_v3/{planet}/{track}_manifest.jsonl"
                manifest.write_text("".join(json.dumps(r) + "\n" for r in self.spec["tracks"][planet][track]), encoding="utf-8")
        recipe = self.root / "recipe.json"
        recipe.write_text(json.dumps(self.spec), encoding="utf-8")
        self.patches = [patch("api.settings.WORKSPACE_ROOT", self.root), patch("planetary_vlm.datasets.orbital.RECOVERY", recipe)]
        for item in self.patches:
            item.start()
        self.settings = Settings(database_path=self.root / ".state/api.sqlite3", _env_file=None)
        self.client = TestClient(create_app(self.settings))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def ensure(self, planet="mars", subset="segmentation"):
        return self.client.post(f"/datasets/builtin/{planet}/ensure", json={"subset": subset})

    def test_local_catalog_is_permanent_read_only_and_registration_is_idempotent(self):
        with patch("api.builtin_datasets.download_dataset", side_effect=AssertionError("No network for local data")):
            catalog = self.client.get("/datasets/builtin").json()
            self.assertEqual([p["planet"] for p in catalog], ["mars", "moon"])
            self.assertEqual(self.client.get("/datasets").json(), [])
            first, second = self.ensure(), self.ensure()
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["dataset"]["id"], second.json()["dataset"]["id"])
        self.assertEqual(first.json()["dataset"]["provenance"]["split_counts"], {"test": 1, "val": 1})
        self.assertEqual((self.root / self.spec["tracks"]["mars"]["segmentation"][0]["image"]).read_bytes(), self.pixels)

    def test_height_metadata_is_preserved_as_dem_without_leaking_into_requests(self):
        response = self.ensure("mars", "height")
        self.assertEqual(response.status_code, 200, response.text)
        dataset = response.json()["dataset"]
        rows = [json.loads(line) for line in Path(dataset["manifest_path"]).read_text().splitlines()]
        self.assertEqual(rows[0]["dem"], rows[0]["height_map"])
        self.assertNotIn("mask", rows[0])
        self.assertEqual(dataset["task_id"], "elevation_map_reading")

    def test_changed_local_targets_are_not_registered_or_overwritten(self):
        row = self.spec["tracks"]["mars"]["segmentation"][0]
        path = self.root / row["mask"]
        path.write_bytes(b"changed")
        response = self.ensure()
        self.assertEqual(response.status_code, 422, response.text)
        self.assertIn("Changed orbital asset", response.text)
        self.assertEqual(path.read_bytes(), b"changed")
        self.assertEqual(self.client.get("/datasets").json(), [])

    def test_missing_assets_use_restore_then_register_the_same_frozen_selection(self):
        row = self.spec["tracks"]["mars"]["segmentation"][0]
        (self.root / row["image"]).unlink()
        with patch("planetary_vlm.datasets.orbital.materialize", return_value={"image": self.pixels, "mask": self.pixels}) as loader:
            response = self.ensure()
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(loader.call_count, 1)
        job = self.client.get(f"/datasets/builtin/jobs/{response.json()['job_id']}").json()
        self.assertEqual(job["status"], "ready", job)
        self.assertEqual(job["completed"], 2)
        self.assertEqual((self.root / row["image"]).read_bytes(), self.pixels)
        self.assertEqual(self.ensure().status_code, 200)

    def test_restore_rejects_wrong_source_bytes_and_reports_failure(self):
        row = self.spec["tracks"]["moon"]["segmentation"][0]
        target = self.root / row["image"]
        target.unlink()
        with patch("planetary_vlm.datasets.orbital.materialize", return_value={"image": b"wrong", "mask": self.pixels}):
            response = self.ensure("moon", "imp")
        job = self.client.get(f"/datasets/builtin/jobs/{response.json()['job_id']}").json()
        self.assertEqual(job["status"], "failed")
        self.assertIn("SHA-256", job["error"])
        self.assertFalse(target.exists())
        self.assertEqual(self.client.get("/datasets").json(), [])

    def test_absent_manifest_imp_restore_does_not_fetch_other_lunar_tasks(self):
        manifest = self.root / "data_orbital/orbital_300_v3/moon/segmentation_manifest.jsonl"
        manifest.unlink()
        imp, other = self.spec["tracks"]["moon"]["segmentation"]
        (self.root / imp["image"]).unlink()
        (self.root / other["image"]).unlink()
        with patch("planetary_vlm.datasets.orbital.materialize", return_value={"image": self.pixels, "mask": self.pixels}) as loader:
            response = self.ensure("moon", "imp")
        self.assertEqual(loader.call_count, 1)
        job = self.client.get(f"/datasets/builtin/jobs/{response.json()['job_id']}").json()
        self.assertEqual(job["status"], "ready", job)
        self.assertTrue(inspect_subset(self.root, "moon", "imp")["ready"])
        self.assertFalse(inspect_subset(self.root, "moon", "segmentation")["ready"])

    def test_unknown_subset_and_protected_asset_are_rejected(self):
        self.assertEqual(self.ensure("mars", "imp").status_code, 422)
        self.assertEqual(self.ensure("earth").status_code, 422)
        manifest = self.root / "data_orbital/orbital_300_v3/mars/segmentation_manifest.jsonl"
        row = {**self.spec["tracks"]["mars"]["segmentation"][0], "image": "data_rover/forbidden.png"}
        manifest.write_text(json.dumps(row) + "\n")
        with patch("planetary_vlm.datasets.orbital.materialize", side_effect=AssertionError("Must reject before acquisition")):
            self.assertEqual(self.ensure().status_code, 422)


if __name__ == "__main__":
    unittest.main()

"""Offline lifecycle/CLI fixtures, not scientific dataset results."""
from contextlib import redirect_stdout, redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from planetary_vlm.cli import main
from planetary_vlm.datasets import BaseData, MarsData, LuneData
from planetary_vlm.download_dataset import download_dataset
from planetary_vlm.prepare_dataset import prepare_dataset


class DatasetLifecycleTests(unittest.TestCase):
    def test_base_is_abstract_and_paths_are_instance_local(self):
        with self.assertRaises(TypeError):
            BaseData()
        with tempfile.TemporaryDirectory() as directory:
            left, right = MarsData(Path(directory)/"mars"), LuneData(Path(directory)/"moon")
            self.assertNotEqual(left.root, right.root)
            self.assertEqual(left.raw, left.root/"raw")

    def test_versions_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = MarsData(Path(directory)/"data")
            existing = Path(directory)/"existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError):
                dataset.new_version(existing)
            with self.assertRaises(ValueError):
                dataset.new_version(dataset.root)

    def test_imports_and_cli_help_do_not_load_heavy_backends(self):
        root = Path(__file__).resolve().parents[1]
        code = (f"import sys; sys.path.insert(0,{str(root/'src')!r}); "
                "from planetary_vlm.datasets import MarsData, LuneData; "
                "assert not set(('numpy','cv2','torch','PIL')).intersection(sys.modules)")
        subprocess.run([sys.executable, "-I", "-c", code], cwd=tempfile.gettempdir(), check=True)

    def test_download_dispatch_does_not_fetch_other_domain(self):
        with patch.object(MarsData, "download", return_value={"fixture": True}) as mars, patch.object(LuneData, "download") as moon:
            result = download_dataset(planet="mars", mars_archive=True)
            mars.assert_called_once_with(archive=True)
            moon.assert_not_called()
            self.assertTrue(result["mars"]["fixture"])

    def test_invalid_download_flags_fail_before_network(self):
        with patch.object(MarsData, "download_stereo_sources") as network:
            with self.assertRaises(ValueError):
                download_dataset(planet="mars", stereo_stage="catalog", images=True)
            with self.assertRaises(ValueError):
                download_dataset(planet="moon", mars_archive=True)
            network.assert_not_called()

    def test_moon_missing_semantic_gt_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"version"
            with self.assertRaisesRegex(ValueError, "semantic"):
                LuneData(directory).prepare(output=output)
            self.assertFalse(output.exists())

    def test_cli_calls_shared_prepare_entrypoint(self):
        with patch("planetary_vlm.prepare_dataset.prepare_dataset", return_value="fixture") as build, redirect_stdout(io.StringIO()):
            self.assertEqual(main(["dataset", "prepare", "--planet", "mars", "--root", "custom", "--output", "version"]), 0)
            build.assert_called_once_with(planet="mars", root="custom", samples=None, specification=None, output="version", depth_index=None)

    def test_cli_rejects_missing_lunar_annotations_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            destination = Path(directory)/"moon_v1"
            self.assertEqual(main(["dataset", "prepare", "--planet", "moon", "--output", str(destination)]), 2)
            self.assertFalse(destination.exists())

    def test_pilot_existing_version_rejected_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                MarsData(directory).verify_pilot_pairs(output=directory)

    def test_packaged_default_mapping_matches_repository_protocol(self):
        import tomllib
        root = Path(__file__).resolve().parents[1]
        configured = tomllib.loads((root/"configs/datasets/mars_bench_clean_v1.toml").read_text())
        packaged = tomllib.loads((root/"src/planetary_vlm/datasets/_mars/semantic.toml").read_text())
        self.assertEqual(configured, packaged)


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Optional image dependency")
class DatasetAssemblyTests(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        root.mkdir()
        Image.new("RGB", (4, 4), (80, 90, 100)).save(root/"image.png")
        Image.new("L", (4, 4), 3).save(root/"mask.png")
        index = root/"samples.jsonl"
        index.write_text(json.dumps({"sample_id": "software-fixture", "image_path": "image.png", "mask_path": "mask.png", "group_id": "fixture-group", "split": "test", "license": "fixture-only", "source_url": "https://example.invalid/fixture"})+"\n", encoding="utf-8")
        specification = root/"native.toml"
        specification.write_text('ignore_labels=[]\nroi=[0.0,0.0,1.0,1.0]\nmin_valid_fraction=1.0\ndominant_min_fraction=0.6\nrock_min_pixels=1\nrock_classes=["ROCK"]\ntasks=["rock_presence"]\nquestion_policy="one_per_image"\n[label_to_class]\n3="ROCK"\n', encoding="utf-8")
        return index, specification

    def test_both_domains_prepare_offline_and_keep_targets_separate(self):
        for planet in ("mars", "moon"):
            with self.subTest(planet=planet), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                samples, specification = self.fixture(root/"source")
                with patch("urllib.request.urlopen", side_effect=AssertionError("Prepare must not download")):
                    output = prepare_dataset(planet=planet, root=root/"data", samples=samples, specification=specification, output=root/"dataset_v1")
                manifest = json.loads((output/"manifest.json").read_text())
                self.assertEqual(manifest[planet]["admitted_photographs"], 1)
                self.assertEqual(manifest[planet]["requests"], f"{planet}/test/requests.jsonl")
                requests = [json.loads(line) for line in (output/manifest[planet]["requests"]).read_text().splitlines()]
                self.assertNotIn("answer", requests[0])
                self.assertNotIn("mask_path", requests[0])
                self.assertEqual(manifest[planet]["depth_maps"], 0)
                if planet == "moon":
                    self.assertNotIn("mars", manifest)
                    self.assertEqual(manifest["moon"]["license"], ["fixture-only"])
                with self.assertRaises(FileExistsError):
                    prepare_dataset(planet=planet, root=root/"data", samples=samples, specification=specification, output=output)


@unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("PIL"), "Optional stereo dependencies")
class MarsPathTests(unittest.TestCase):
    def test_metadata_search_uses_custom_root_not_repository_assets(self):
        left_id = "0703ML0029750010304398E01_DXXX"
        right_id = "0703MR0029750000402286E01_DXXX"
        def model(sid):
            return {"product_id": sid, "source_image": sid+".IMG", "C": [0,0,0] if sid == left_id else [.265,0,0], "A": [0,0,1], "O": [0,0,1], "H": [100,0,50], "V": [0,100,50], "R": [0,0,0], "frame": "fixture", "frame_indices": {"site": "1"}, "frame_solution": "fixture", "axes": {"Band": 3, "Sample": 100, "Line": 100}, "start_date_time": "2020-01-01T00:00:00Z"}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            mars = MarsData(directory)
            catalog = mars.raw/"mastcam_partner_search_v1"
            catalog.mkdir(parents=True)
            names = [x.lower()+".xml" for x in (left_id, right_id)]
            (catalog/"catalog.json").write_text(json.dumps([{"sol":"0703", "url":"https://example.invalid/", "label_names": names}]))
            mars.samples.parent.mkdir(parents=True)
            (mars.samples.parent/"provenance.jsonl").write_text(json.dumps({"sample_id":"fixture", "mission_image_id":right_id})+"\n")
            mars.search_pairs("requests")
            requests = json.loads((catalog/"label_requests.json").read_text())
            (catalog/"labels").mkdir()
            for row in requests:
                (catalog/"labels"/row["name"]).write_text("fixture XML handled by parser stub")
            (catalog/"label_acquisition.json").write_text(json.dumps([dict(row, status="downloaded") for row in requests]))
            with patch("planetary_vlm.datasets._mars.partners.inspect_label", side_effect=lambda path, output, **opts: copy.deepcopy(model(path.stem.upper()))):
                report_path = mars.search_pairs("audit")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["benchmark_frames"], 1)
            self.assertEqual(report["benchmark_frames_with_candidates"], 1)
            self.assertTrue(report["acquisition_complete"])
            self.assertTrue(report_path.is_relative_to(mars.root))

"""CPU-only IBM fixtures; not evidence that weights load or that GPU works."""
import ast
from contextlib import nullcontext
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from planetary_vlm import colab, colab_ibm as ibm
from benchmark_services.ibm import crater
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS
from planetary_vlm.inference.runner import file_hash
REAL_ACQUIRE_ASSETS = ibm.acquire_assets

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = Image = None


class CraterRulesTests(unittest.TestCase):
    def test_no_detections_is_a_predicted_negative_not_missing_ground_truth(self):
        result = crater.readout([], [], [], 256, 256)
        self.assertEqual(list(result["answers"].values()), ["NO", "NO", "NONE", "TIE"])

    def test_geometry_filters_and_threshold_are_frozen(self):
        result = crater.readout([[100,100,160,160], [1,1,11,11], [0,5,70,70], [180,180,230,230]],
                                [.5, .9, .99, .499], [1,1,1,1], 256, 256)
        self.assertEqual(result["eligible_count"], 1)
        self.assertEqual(list(result["answers"].values()), ["YES", "YES", "5", "RIGHT"])
        self.assertEqual(len(result["detections"]), 4)

    def test_size_tie_and_half_count(self):
        result = crater.readout([[5,5,30,30], [190,190,215,215]], [.6,.7], [1,1], 256,256)
        self.assertEqual(list(result["answers"].values()), ["YES", "NO", "TIE", "TIE"])

    def test_malformed_output_fails(self):
        for boxes, scores, labels in (([[1,1,20,20]], [], [1]), ([[1,1,20,20]], [float("nan")], [1]),
            ([[1,1,20,20]], [.9], [0]), ([[1,1,260,20]], [.9], [1]), ([[1,1,1,20]], [.9], [1])):
            with self.subTest(boxes=boxes, scores=scores, labels=labels), self.assertRaises(ValueError):
                crater.readout(boxes,scores,labels,256,256)

    def test_nac_finetuned_encoder_must_be_present(self):
        crater.validate_model_keys(["criterion.weight"], ["metric_state"])
        for missing, extra in ((["model.torchvision_model.backbone.backbone.model.encoder.0.weight"], []),
                               (["model.head.weight"], []), ([], ["model.foreign"])):
            with self.assertRaisesRegex(ValueError, "Incomplete NAC"):
                crater.validate_model_keys(missing, extra)

    def test_wrong_protocol_rejected_without_torch(self):
        for opts in ({}, {"input_protocol": crater.PROTOCOL, "code_revision": "wrong"}):
            with self.assertRaises(ValueError):
                crater.IBMCraterDetector(opts).preflight()

    def test_config_rejects_wrong_head_or_normalization(self):
        config = {"model": {"class_path": "terratorch_integration.LunarObjectDetectionTask", "init_args": {
            "freeze_backbone": False, "model_args": {"framework": "faster-rcnn", "backbone": "ni_lfm_v1_base",
            "backbone_modalities": ["nac"], "backbone_patch_size": 8, "num_classes": 2,
            "in_channels": 1, "framework_min_size": 256, "framework_max_size": 256}}},
            "data": {"init_args": {"nac_norm_mean": crater.MEAN, "nac_norm_std": crater.STD, "image_size": 256}}}
        args = crater.validate_config(config)
        args["model_args"]["backbone"] = "changed"
        self.assertEqual(config["model"]["init_args"]["model_args"]["backbone"], "ni_lfm_v1_base")
        config["data"]["init_args"]["nac_norm_std"] = .12
        with self.assertRaisesRegex(ValueError, "normalization"):
            crater.validate_config(config)


@unittest.skipIf(Image is None, "Optional NumPy/Pillow missing")
class IBMOfflineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)/"workspace"
        self.persist = Path(self.temporary.name)/"drive"
        requests, provenance = [], []
        for i, sid in enumerate(ibm.SMOKE_IDS):
            path = self.root/"data_orbital/colab_inputs/images"/sid/"clean.png"
            path.parent.mkdir(parents=True)
            Image.fromarray(np.full((256,256), i*64, dtype=np.uint8)).save(path)
            for task, answers in zip(TASKS, ANSWERS):
                rid = sid+"__"+task+"__clean"
                requests.append({"request_id": rid, "image_paths": [path.relative_to(self.root).as_posix()],
                    "allowed_answers": answers, "prompt": "Image-only fixture", "modality_paths": []})
                provenance.append({"request_id": rid, "sample_id": sid, "planet": "moon",
                    "task_id": task, "track": "segmentation", "condition": "clean", "evaluation_eligible": True,
                    "source_split": "val", "source_dataset": "synthetic-fixture", "image_sha256": file_hash(path)})
        for name, rows in (("requests", requests), ("provenance", provenance)):
            (self.root/"data_orbital/colab_inputs"/(name+".jsonl")).write_text(
                "".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        self.events = []
        events = self.events
        class FixtureDetector:
            def __init__(self, options): pass
            def initialize(self):
                events.append("initialize")
                return {"synthetic_fixture": True}
            def detect(self, path):
                events.append(Path(path).parent.name)
                result = crater.readout([], [], [], 256,256)
                return {**result, "preprocessing": {"native_width": 256, "native_height": 256}}
            def close(self): events.append("close")
        self.detector_class = FixtureDetector
        for target, value in (("verify_bundle", {"bundle_sha256": "fixture-only"}),
                              ("environment", {"synthetic_fixture": True}), ("acquire_assets", {})):
            mocker = patch.object(ibm, target, return_value=value)
            mocker.start()
            self.addCleanup(mocker.stop)
        mocker = patch.object(ibm, "IBMCraterDetector", self.detector_class)
        mocker.start()
        self.addCleanup(mocker.stop)

    def run_job(self, *, smoke=True, resume=False, session="session1"):
        with redirect_stdout(StringIO()):
            return ibm.run_job(self.root,self.persist,"fixture_run",session,smoke=smoke,resume=resume)

    def test_preprocessing_and_native_coordinate_rules(self):
        path = self.root/"data_orbital/colab_inputs/images"/ibm.SMOKE_IDS[1]/"clean.png"
        image, record = crater.prepare_png(path)
        self.assertEqual(image.shape, (1,1,256,256))
        self.assertEqual(image.dtype, np.float32)
        self.assertAlmostEqual(float(image[0,0,0,0]), (64/255-crater.MEAN)/crater.STD, places=6)
        self.assertEqual(record["input_tensor_sha256"], sha256(image.tobytes()).hexdigest())
        result = crater.readout([[30,30,90,90]], [.8], [1], 768,768)
        self.assertEqual(result["eligible_count"], 1)
        self.assertEqual(result["answers"][TASKS[2]], "1")

    def test_detect_calls_publisher_nms_and_rescales_before_geometry(self):
        from unittest.mock import MagicMock
        path = self.root/"large.png"
        Image.fromarray(np.zeros((768,768), dtype=np.uint8)).save(path)
        class Tensor:
            def __init__(self, value): self.value = value
            def to(self, **kwargs): return self
            def float(self): return self
            def cpu(self): return self
            def tolist(self): return self.value
        raw = [{"boxes": Tensor([[100.,100.,160.,160.]]), "scores": Tensor([.8]), "labels": Tensor([1])}]
        adapter = crater.IBMCraterDetector({"input_protocol": crater.PROTOCOL})
        adapter.torch = SimpleNamespace(from_numpy=lambda x: Tensor(x), float32="fixture-float32", inference_mode=nullcontext)
        adapter.model = MagicMock(return_value=SimpleNamespace(output=raw))
        adapter.model.apply_nms_batch.return_value = raw
        result = adapter.detect(path)
        self.assertEqual(result["detections"][0]["bbox_xyxy"], [300.,300.,480.,480.])
        self.assertEqual(result["answers"][TASKS[2]], "5")
        self.assertEqual(adapter.model.call_count, 1)
        adapter.model.apply_nms_batch.assert_called_once_with(raw,1)

    def test_input_grouping_does_not_need_masks_or_targets(self):
        images = ibm.select_inputs(self.root, smoke=True)
        self.assertEqual([r["sample_id"] for r in images], list(ibm.SMOKE_IDS))
        self.assertTrue(all(set(r["requests"]) == set(TASKS) for r in images))

    def test_corrupt_image_and_duplicate_question_rejected(self):
        path = self.root/"data_orbital/colab_inputs/requests.jsonl"
        with path.open("a") as stream:
            stream.write(path.read_text().splitlines()[0]+"\n")
        with self.assertRaises(ValueError):
            ibm.select_inputs(self.root)

    def test_assets_are_never_implicitly_downloaded(self):
        with self.assertRaisesRegex(ValueError,"explicitly"):
            REAL_ACQUIRE_ASSETS(self.root)

    def test_run_requires_successful_current_smoke(self):
        with self.assertRaisesRegex(ValueError, "successful IBM smoke"):
            self.run_job(smoke=False)
        self.assertEqual(self.events, [])
        self.run_job()
        with self.assertRaisesRegex(ValueError, "successful IBM smoke"):
            self.run_job(smoke=False, session="session2")

    def test_single_load_checkpoints_answers_and_durable_resume(self):
        smoke = self.run_job()
        self.assertEqual(smoke["answers"], 8)
        self.assertEqual(self.events, ["initialize", *ibm.SMOKE_IDS, "close"])
        self.events.clear()
        result = self.run_job(smoke=False)
        destination = Path(result["result_dir"])
        rows = [json.loads(line) for line in (destination/"predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertFalse(result["scientific_score_available"])
        self.assertEqual(len(list((Path(result["persistent_dir"])/"overlays").glob("*.png"))),2)
        self.events.clear()
        self.run_job(smoke=False, resume=True)
        self.assertEqual(self.events, ["close"])
        with zipfile.ZipFile(self.persist/"fixture_run_results.zip") as archive:
            self.assertEqual(sum("/detections/" in p for p in archive.namelist()), 2)

    def test_failure_never_issues_smoke_permission_and_keeps_error(self):
        with patch.object(self.detector_class, "initialize", side_effect=RuntimeError("fixture load failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture load failure"):
                self.run_job()
        self.assertFalse((self.root/"ibm_smoke.json").exists())
        self.assertEqual(self.events, ["close"])
        with zipfile.ZipFile(self.persist/"fixture_run_smoke_session1_results.zip") as archive:
            self.assertTrue(any(name.endswith("/error.json") for name in archive.namelist()))

    def test_modified_detections_refuse_resume(self):
        self.run_job()
        result = self.run_job(smoke=False)
        for folder in (result["result_dir"], result["persistent_dir"]):
            path = Path(folder)/"detections"/(ibm.SMOKE_IDS[0]+".json")
            row = json.loads(path.read_text())
            row["answers"][TASKS[0]] = "YES"
            path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "detections changed"):
            self.run_job(smoke=False, resume=True)

    def test_missing_committed_detection_refuses_resume(self):
        self.run_job()
        result = self.run_job(smoke=False)
        for folder in (result["result_dir"], result["persistent_dir"]):
            path = Path(folder)/"detections"/(ibm.SMOKE_IDS[0]+".json")
            path.rename(path.with_suffix(".fixture-hidden"))
        with self.assertRaisesRegex(ValueError,"Incomplete IBM checkpoint transaction"):
            self.run_job(smoke=False,resume=True)

    def test_resume_restores_from_drive_without_loading_completed_images(self):
        self.run_job()
        result = self.run_job(smoke=False)
        local = Path(result["result_dir"])
        local.rename(local.with_name("fixture-old-local"))
        self.events.clear()
        self.run_job(smoke=False,resume=True)
        self.assertTrue((local/"predictions.jsonl").is_file())
        self.assertEqual(self.events,["close"])

    def test_notebook_schema_code_and_bundle_bootstrap_offline(self):
        if not (ROOT/'notebooks/ibm_craters_colab.ipynb').is_file() or not (ROOT/'outputs/colab/ibm_craters_v1.zip').is_file():
            self.skipTest('Generated legacy notebook/ZIP intentionally removed')
        import nbformat
        import importlib.util
        # A user's executed notebook is an experiment log, not a pristine
        # distribution template. Validate its schema WITHOUT clearing outputs;
        # assert safe defaults on the actual in-memory export factory instead.
        saved = nbformat.read(ROOT/"notebooks/ibm_craters_colab.ipynb", as_version=4)
        nbformat.validate(saved)
        spec = importlib.util.spec_from_file_location("ibm_colab_template_fixture", ROOT/"notebooks/build_ibm_colab.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        bundle_zip = ROOT/"outputs/colab/ibm_craters_v1.zip"
        notebook = builder.notebook(sha256(bundle_zip.read_bytes()).hexdigest())
        nbformat.validate(notebook)
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        for cell in code_cells:
            ast.parse(cell.source)
            self.assertEqual(cell.outputs, [])
        self.assertEqual(len(code_cells),7)
        self.assertIn("DOWNLOAD_WEIGHTS = False", code_cells[0].source)
        self.assertIn("RUN_BENCHMARK = False", code_cells[-1].source)
        source = next(c.source for c in code_cells if c.id == "unpack")
        target = Path(self.temporary.name)/"unpacked"
        context = dict(Path=Path, zipfile=zipfile, json=json, hashlib=__import__("hashlib"),
                       stat=__import__("stat"), BUNDLE_ZIP=bundle_zip, WORKSPACE=target, PERSIST=self.persist)
        with redirect_stdout(StringIO()):
            exec(compile(source,"ibm-unpack-cell","exec"),context)
        manifest = colab.verify_bundle(target)
        self.assertEqual(manifest["selected_images"],100)
        self.assertEqual(manifest["requests"],400)
        self.assertEqual(len(ibm.select_inputs(target)),100)
        with zipfile.ZipFile(bundle_zip) as archive:
            inputs = [p for p in archive.namelist() if p.startswith("data_orbital/")]
            self.assertEqual(len(inputs),102)
            self.assertFalse(any("mask" in p or "target" in p or p.endswith(".npy") for p in inputs))
            self.assertIn("requirements/ibm-imp.txt",archive.namelist())
            self.assertIn("requirements/ibm-crater-colab.txt",archive.namelist())
            self.assertNotIn("requirements/spacellava.txt",archive.namelist())
            for name in archive.namelist():
                if name.endswith(".py"):
                    ast.parse(archive.read(name).decode("utf-8-sig"))

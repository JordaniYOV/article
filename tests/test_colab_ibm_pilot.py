"""Synthetic CPU fixtures only: bounded cohort, GT isolation and saved-run QA."""

import ast
from contextlib import redirect_stdout
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from planetary_vlm import colab, colab_ibm as ibm, colab_ibm_pilot as pilot
from planetary_vlm.datasets.crater_reference import TASKS, ANSWERS
from planetary_vlm.io import read_jsonl, write_jsonl

try:
    from PIL import Image
except ImportError:
    Image = None


@unittest.skipIf(Image is None, "Optional Pillow missing")
class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ibp_")
        self.addCleanup(self.tmp.cleanup)
        self.root, self.persist = Path(self.tmp.name)/"w", Path(self.tmp.name)/"d"
        self.pilot_ids = [f"{planet}_pilot_{i:03d}" for planet in ("moon", "mars") for i in range(7)]
        self.selection = {"schema": pilot.SCHEMA, "selection_version": "ibm-val14-source-group-v1",
            "pilot_sample_ids": self.pilot_ids, "smoke_sample_ids": list(ibm.SMOKE_IDS)}
        self.requests, self.provenance = [], []
        for sid in self.pilot_ids+list(ibm.SMOKE_IDS):
            name = f"data_orbital/colab_inputs/images/{sid}/clean.png"
            path = self.root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (32,32), 64).save(path)
            for task, allowed in zip(TASKS, ANSWERS):
                rid = sid+"__"+task+"__clean"
                self.requests.append({"request_id": rid, "image_paths": [name], "allowed_answers": list(allowed),
                    "prompt": "Synthetic CPU fixture, not a scientific question", "max_new_tokens": 16,
                    "do_sample": False, "modality_paths": []})
                self.provenance.append({"request_id": rid, "sample_id": sid, "task_id": task,
                    "planet": sid.split("_")[0], "source_dataset": "synthetic-fixture",
                    "source_split": "val", "scene_group_id": "group-"+sid, "track": "segmentation",
                    "condition": "clean", "evaluation_eligible": True,
                    "image_sha256": sha256(path.read_bytes()).hexdigest()})
        self.seal()
        self.events = []
        events = self.events
        class Detector:
            def __init__(self, options): pass
            def initialize(self):
                events.append("initialize")
                return {"synthetic_fixture": True}
            def detect(self, path):
                events.append(Path(path).parent.name)
                return {**ibm.IBMCraterDetector_fixture_readout(), "preprocessing": {"native_width": 32, "native_height": 32}}
            def close(self): events.append("close")
        # Keep a reference to the real geometry function, not model inference.
        from benchmark_services.ibm.crater import readout
        mocker = patch.object(ibm, "IBMCraterDetector_fixture_readout", lambda: readout([], [], [], 32, 32), create=True)
        mocker.start()
        self.addCleanup(mocker.stop)
        for target, value in (("environment", {"gpu": "SYNTHETIC_CPU_NOT_A_GPU_RESULT"}),
            ("acquire_assets", {}), ("IBMCraterDetector", Detector)):
            mocker = patch.object(ibm, target, return_value=value) if target != "IBMCraterDetector" else patch.object(ibm, target, value)
            mocker.start()
            self.addCleanup(mocker.stop)

    def seal(self):
        base = self.root/"data_orbital/colab_inputs"
        for name, rows in (("requests", self.requests), ("provenance", self.provenance)):
            (base/(name+".jsonl")).write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
        names = {row["image_paths"][0] for row in self.requests}
        names.update(("data_orbital/colab_inputs/requests.jsonl", "data_orbital/colab_inputs/provenance.jsonl"))
        record = {"schema": colab.SCHEMA, "protocol": {"inference_cohort": self.selection},
            "files_sha256": {name: sha256((self.root/name).read_bytes()).hexdigest() for name in names},
            "selected_images": 16, "requests": 64}
        record["bundle_sha256"] = colab.digest(record)
        (self.root/"colab_bundle.json").write_text(json.dumps(record), encoding="utf-8")
        self.bundle = record

    def run_pilot(self, *, smoke=False, resume=False, session="s1"):
        with redirect_stdout(StringIO()):
            return pilot.run_pilot(self.root, self.persist, "fixture", session, smoke=smoke, resume=resume)

    def reference_kit(self):
        folder = Path(self.tmp.name)/"r"
        folder.mkdir()
        selected = [row for row in self.requests if row["request_id"].split("__")[0] in self.pilot_ids]
        selected = [{**row, "image_paths": [str(self.root/row["image_paths"][0])]} for row in selected]
        targets = [{"request_id": row["request_id"], "source_sample_id": row["request_id"].split("__")[0],
            "group_id": "group-"+row["request_id"].split("__")[0], "derivation_version": "synthetic-fixture",
            "task_id": row["request_id"].split("__")[1], "condition_id": "clean",
            "answer": dict(zip(TASKS, ("NO", "NO", "NONE", "TIE")))[row["request_id"].split("__")[1]],
            "reference_status": "CANDIDATE_REFERENCE"} for row in selected]
        for name, rows in (("requests.jsonl", selected), ("candidate_targets.jsonl", targets), ("source_candidates.jsonl", targets)):
            write_jsonl(folder/name, rows)
        record = {"schema": "ibm-val14-local-references-v1", "input_bundle_sha256": self.bundle["bundle_sha256"],
            "pilot_sample_ids": self.pilot_ids, "image_sha256": {row["sample_id"]: row["image_sha256"]
                for row in self.provenance if row["sample_id"] in self.pilot_ids},
            "files_sha256": {name: sha256((folder/name).read_bytes()).hexdigest() for name in (
                "requests.jsonl", "candidate_targets.jsonl", "source_candidates.jsonl")}}
        record["manifest_sha256"] = colab.digest(record)
        (folder/"reference_manifest.json").write_text(json.dumps(record), encoding="utf-8")
        return folder

    def test_frozen_cohort_is_14_new_val_groups_plus_two_controls(self):
        _, value = pilot.cohort(self.root)
        self.assertEqual(value["pilot_sample_ids"], self.pilot_ids)
        self.assertFalse(set(self.pilot_ids) & set(ibm.SMOKE_IDS))

    def test_test_split_is_rejected_before_any_model_call(self):
        self.provenance[0]["source_split"] = "test"
        self.seal()
        with self.assertRaisesRegex(ValueError, "publisher test/train"):
            pilot.cohort(self.root)
        self.assertEqual(self.events, [])

    def test_related_groups_cannot_repeat(self):
        sid = self.pilot_ids[1]
        for row in self.provenance:
            if row["sample_id"] == sid:
                row["scene_group_id"] = "group-"+self.pilot_ids[0]
        self.seal()
        with self.assertRaisesRegex(ValueError, "groups must not repeat"):
            pilot.cohort(self.root)

    def test_cohort_cannot_contain_reference_metadata(self):
        self.selection["expected_presence"] = "PRIVATE_GROUND_TRUTH"
        self.seal()
        with self.assertRaisesRegex(ValueError, "sealed IBM"):
            pilot.cohort(self.root)

    def test_no_pilot_without_current_session_smoke(self):
        with self.assertRaisesRegex(ValueError, "successful IBM smoke"):
            self.run_pilot()
        self.assertEqual(self.events, [])

    def test_smoke_and_pilot_are_disjoint_and_resume_skips_completed_images(self):
        self.assertEqual(self.run_pilot(smoke=True)["images"], 2)
        self.events.clear()
        result = self.run_pilot()
        self.assertEqual((result["images"], result["answers"]), (14,56))
        self.assertEqual(set(self.events)-{"initialize", "close"}, set(self.pilot_ids))
        self.assertEqual(self.events.count("initialize"), 1)
        self.assertEqual(len(read_jsonl(Path(result["result_dir"])/"predictions.jsonl")),56)
        self.events.clear()
        self.run_pilot(resume=True)
        self.assertEqual(self.events, ["close"])
        with self.assertRaisesRegex(ValueError, "successful IBM smoke"):
            self.run_pilot(resume=True, session="new-session")

    def test_shared_runner_rejects_unknown_or_duplicate_explicit_ids(self):
        self.run_pilot(smoke=True)
        for values in ([self.pilot_ids[0]]*2, ["foreign"], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ibm.run_job(self.root,self.persist,"other","s1",sample_ids=values)

    def test_local_evaluation_requires_explicit_provisional_opt_in(self):
        self.run_pilot(smoke=True)
        self.run_pilot()
        refs = self.reference_kit()
        zip_path = self.persist/"fixture_results.zip"
        a = pilot.score_results(zip_path, refs, Path(self.tmp.name)/"unscored.json")
        self.assertIsNone(a["overall"]["accuracy"])
        self.assertEqual(a["overall"]["n_unscored_reference"],56)
        b = pilot.score_results(zip_path, refs, Path(self.tmp.name)/"provisional.json", provisional=True)
        self.assertTrue(b["complete"])
        self.assertEqual(b["overall"]["n_scored"],56)
        self.assertFalse(b["scientific_result"])
        # Perfect matches are synthetic pipeline fixtures, NOT model quality.
        self.assertTrue(all(row["reference_status"] == "CANDIDATE_REFERENCE" for row in read_jsonl(refs/"candidate_targets.jsonl")))

    def test_smoke_zip_cannot_be_scored_as_the_pilot(self):
        self.run_pilot(smoke=True)
        with self.assertRaisesRegex(ValueError, "Not this frozen pilot"):
            pilot.score_results(self.persist/"fixture_smoke_s1_results.zip", self.reference_kit(),
                Path(self.tmp.name)/"bad.json", provisional=True)

    def test_changed_local_reference_refuses_evaluation(self):
        self.run_pilot(smoke=True)
        self.run_pilot()
        refs = self.reference_kit()
        (refs/"candidate_targets.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference file changed"):
            pilot.score_results(self.persist/"fixture_results.zip", refs, Path(self.tmp.name)/"bad.json")

    def test_hash_bound_tiff_under_png_name_is_decoded_without_source_changes(self):
        sid = self.pilot_ids[0]
        path = self.root/f"data_orbital/colab_inputs/images/{sid}/clean.png"
        Image.new("L",(32,32),64).save(path,format="TIFF")
        checksum = sha256(path.read_bytes()).hexdigest()
        for row in self.provenance:
            if row["sample_id"] == sid:
                row["image_sha256"] = checksum
        self.seal()
        self.run_pilot(smoke=True)
        self.run_pilot()
        result = pilot.score_results(self.persist/"fixture_results.zip",self.reference_kit(),
            Path(self.tmp.name)/"tiff.json",provisional=True)
        self.assertEqual(result["overall"]["n_scored"],56)
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),checksum)


class PilotExportTests(unittest.TestCase):
    @unittest.skipIf(Image is None, "Optional Pillow missing")
    def test_shared_export_explicit_source_filter_and_cohort_allowlist(self):
        from test_colab import ColabTests
        fixture = ColabTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        values = ["moon_segmentation_001", "mars_segmentation_001"]
        cohort = {"schema": "fixture", "selection_version": "fixture",
            "pilot_sample_ids": [values[0]], "smoke_sample_ids": [values[1]]}
        with redirect_stdout(StringIO()):
            info = colab.export_bundle(fixture.root, fixture.zip_path, conditions=("clean",),
                source_ids=values, inference_cohort=cohort)
        self.assertEqual(info["selected_images"],2)
        self.assertEqual(info["excluded"]["outside_explicit_cohort"],2)
        with zipfile.ZipFile(fixture.zip_path) as z:
            data = json.loads(z.read("colab_bundle.json"))
            self.assertEqual(data["protocol"]["source_ids"], sorted(values))
            self.assertEqual(data["protocol"]["inference_cohort"],cohort)
            self.assertNotIn("PRIVATE_GROUND_TRUTH",z.read("data_orbital/colab_inputs/provenance.jsonl").decode())
        for ids in (["foreign"], values*2, []):
            with self.assertRaises(ValueError):
                colab.export_bundle(fixture.root, fixture.root/"outputs/colab/invalid.zip", conditions=("clean",), source_ids=ids)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            colab.export_bundle(fixture.root, fixture.root/"outputs/colab/invalid.zip", conditions=("clean",),
                source_ids=values, inference_cohort={**cohort, "answer": "NO"})

    def test_real_notebook_bootstrap_and_frozen_selection_are_cpu_verified(self):
        if not (ROOT/'outputs/colab/ibm_val14_v1.zip').is_file():
            self.skipTest('Generated legacy Colab ZIP intentionally removed')
        import nbformat
        spec = importlib.util.spec_from_file_location("ibm_pilot_builder_test", ROOT/"notebooks/build_ibm_pilot.py")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0,str(ROOT/"notebooks"))
        self.addCleanup(sys.path.remove,str(ROOT/"notebooks"))
        spec.loader.exec_module(module)
        first = module.select(ROOT)[2]
        self.assertEqual(module.select(ROOT)[2],first)
        self.assertEqual(len(first["pilot_sample_ids"]),14)
        self.assertFalse(set(first["pilot_sample_ids"]) & set(ibm.SMOKE_IDS))
        locked = [line for line in (ROOT/"requirements/ibm-val14-colab-lock.txt").read_text().splitlines()
            if line and not line.startswith("#")]
        self.assertEqual(len(locked),156)
        self.assertEqual(colab.digest(sorted(locked)),module.FREEZE_SHA256)
        path = ROOT/"outputs/colab/ibm_val14_v1.zip"
        # The user-facing current notebook can advance to the full-run version;
        # exercise the historical pilot factory against its immutable old ZIP.
        document = module.notebook(sha256(path.read_bytes()).hexdigest())
        nbformat.validate(document)
        for cell in document.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])
        self.assertIn("RUN_PILOT = False",next(c.source for c in document.cells if c.id=="pilot"))
        self.assertIn("DOWNLOAD_WEIGHTS = False",next(c.source for c in document.cells if c.id=="parameters"))
        with tempfile.TemporaryDirectory(prefix="ipb_") as tmp:
            import hashlib,stat
            target = Path(tmp)/"w"
            context = dict(Path=Path,zipfile=zipfile,json=json,hashlib=hashlib,stat=stat,
                BUNDLE_ZIP=path,WORKSPACE=target,PERSIST=Path(tmp)/"d")
            with redirect_stdout(StringIO()):
                exec(compile(next(c.source for c in document.cells if c.id=="unpack"),"pilot-unpack","exec"),context)
            _, selection = pilot.cohort(target)
            self.assertEqual(selection,first)
            self.assertEqual(len(ibm.select_inputs(target)),16)
        with zipfile.ZipFile(path) as z:
            data = [name for name in z.namelist() if name.startswith("data_orbital/")]
            self.assertEqual(len(data),18)  # 16 PNGs + requests/provenance, no GT.
            self.assertTrue(all(name.endswith(".png") or name.endswith(("requests.jsonl","provenance.jsonl")) for name in data))
            for name in z.namelist():
                if name.endswith(".py"):
                    ast.parse(z.read(name).decode("utf-8-sig"))


if __name__ == "__main__":
    unittest.main()

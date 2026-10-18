"""CPU-only full-run routing, notebook and evaluation regressions; no real IBM."""
import ast
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from planetary_vlm import colab, colab_ibm_full as full, colab_ibm_pilot as pilot
from planetary_vlm.io import read_jsonl, write_jsonl
import test_colab_ibm_pilot as fixtures


@unittest.skipIf(fixtures.Image is None,"Optional Pillow missing")
class FullTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PilotTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        for planet,count in (("moon",41),("mars",43)):
            template = next(r for r in f.provenance if r["planet"]==planet)
            for index in range(count):
                sid=f"{planet}_extra_{index:03d}"
                path=f.root/f"data_orbital/colab_inputs/images/{sid}/clean.png"
                path.parent.mkdir(parents=True)
                fixtures.Image.new("L",(32,32),64).save(path)
                for request in f.requests[:4]:
                    task=request["request_id"].split("__")[1]
                    rid=sid+"__"+task+"__clean"
                    f.requests.append({**request,"request_id":rid,"image_paths":[path.relative_to(f.root).as_posix()]})
                    f.provenance.append({**template,"request_id":rid,"task_id":task,"sample_id":sid,
                        "scene_group_id":"group-"+sid,"source_split":"test"})
        f.seal()
        record=json.loads((f.root/"colab_bundle.json").read_text())
        record.pop("bundle_sha256")
        record.update(selected_images=100,requests=400,counts={"fixture":100})
        record["protocol"]={"conditions":["clean"],"source_ids":sorted({r["sample_id"] for r in f.provenance})}
        record["bundle_sha256"]=colab.digest(record)
        (f.root/"colab_bundle.json").write_text(json.dumps(record))
        f.bundle=record

    def test_full_cohort_keeps_all_100_and_refuses_pilot_override(self):
        _,images=full.cohort(self.fixture.root)
        self.assertEqual(len(images),100)
        self.assertEqual(sum(r["planet"]=="moon" for r in images),50)
        with self.assertRaises(ValueError):
            pilot.cohort(self.fixture.root)

    def test_full_run_smoke_checkpoint_resume_and_400_provisional_evaluation(self):
        f=self.fixture
        with self.assertRaisesRegex(ValueError,"successful IBM smoke"):
            full.main(["run","--workspace",str(f.root),"--persist",str(f.persist),"--run-id","full","--session-id","s1"])
        with redirect_stdout(StringIO()):
            full.main(["smoke","--workspace",str(f.root),"--persist",str(f.persist),"--run-id","full","--session-id","s1"])
            f.events.clear()
            # Unit-test Drive I/O is batched, while all 101 checkpoint calls are
            # asserted. Per-image real checkpoint semantics have separate runner
            # tests; avoid O(N^2) filesystem copies in this synthetic 100 fixture.
            real_checkpoint=fixtures.ibm.copy_checkpoint
            def bounded_drive_io(source,destination):
                progress=json.loads((source/"progress.json").read_text())
                if progress["completed_images"]==100:
                    real_checkpoint(source,destination)
            with patch.object(fixtures.ibm,"copy_checkpoint",side_effect=bounded_drive_io) as checkpoint:
                full.main(["run","--workspace",str(f.root),"--persist",str(f.persist),"--run-id","full","--session-id","s1"])
                self.assertEqual(checkpoint.call_count,101)
        self.assertEqual(len(f.events)-2,100)
        f.events.clear()
        with redirect_stdout(StringIO()):
            full.main(["run","--workspace",str(f.root),"--persist",str(f.persist),"--run-id","full","--session-id","s1","--resume"])
        self.assertEqual(f.events,["close"])
        refs=Path(f.tmp.name)/"full_refs"
        refs.mkdir()
        by_task={task:{"request_id":"template","source_sample_id":"template","group_id":"template",
            "derivation_version":"synthetic-fixture","task_id":task,"condition_id":"clean",
            "answer":answer,"reference_status":"CANDIDATE_REFERENCE"}
            for task,answer in zip(fixtures.TASKS,("NO","NO","NONE","TIE"))}
        targets=[]
        for request in f.requests:
            sid,task,_=request["request_id"].split("__")
            targets.append({**by_task[task],"request_id":request["request_id"],"source_sample_id":sid,
                "group_id":"group-"+sid,"planet":sid.split("_")[0],"source_dataset":"synthetic-fixture",
                "source_split":"test" if "extra" in sid else "val","prior_ibm_development_scene":"extra" not in sid})
        write_jsonl(refs/"candidate_targets.jsonl",targets)
        write_jsonl(refs/"requests.jsonl",[{**r,"image_paths":[str(f.root/r["image_paths"][0])]} for r in f.requests])
        write_jsonl(refs/"source_candidates.jsonl",targets)
        seal={"schema":"ibm-full100-local-references-v2","input_bundle_sha256":f.bundle["bundle_sha256"],
            "sample_ids":sorted({r["sample_id"] for r in f.provenance}),
            "image_sha256":{r["sample_id"]:r["image_sha256"] for r in f.provenance},
            "files_sha256":{n:sha256((refs/n).read_bytes()).hexdigest() for n in
                ("requests.jsonl","candidate_targets.jsonl","source_candidates.jsonl")}}
        seal["manifest_sha256"]=colab.digest(seal)
        (refs/"reference_manifest.json").write_text(json.dumps(seal))
        out=Path(f.tmp.name)/"full-eval.json"
        with redirect_stdout(StringIO()):
            full.main(["evaluate","--results",str(f.persist/"full_results.zip"),"--reference-dir",str(refs),"--output",str(out),"--provisional"])
        report=json.loads(out.read_text())
        self.assertEqual(report["overall"]["n_scored"],400)
        self.assertEqual(report["requested_images"],100)
        self.assertFalse(report["scientific_result"])
        self.assertEqual(sum(r["n_total"] for r in report["descriptive_planet_split_task_slices"].values()),400)
        self.assertTrue(all(r["reference_status"]=="CANDIDATE_REFERENCE" for r in read_jsonl(refs/"candidate_targets.jsonl")))
        with self.assertRaisesRegex(ValueError,"seal/schema"):
            pilot.score_results(f.persist/"full_results.zip",refs,Path(f.tmp.name)/"wrong.json")


class FullNotebookTests(unittest.TestCase):
    def test_current_factory_has_single_run_switch_and_immutable_pilot_settings(self):
        import nbformat
        sys.path.insert(0,str(ROOT/"notebooks"))
        self.addCleanup(sys.path.remove,str(ROOT/"notebooks"))
        spec=importlib.util.spec_from_file_location("ibm_full_builder_test",ROOT/"notebooks/build_ibm_full.py")
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        nb=module.notebook("fixture-checksum")
        nbformat.validate(nb)
        codes={cell.id:cell.source for cell in nb.cells if cell.cell_type=="code"}
        for cell in nb.cells:
            if cell.cell_type=="code":
                ast.parse(cell.source)
                self.assertEqual(cell.outputs,[])
        self.assertNotIn("RUN_BENCHMARK",codes["parameters"])
        self.assertIn("RUN_BENCHMARK = False",codes["run"])
        self.assertIn("DOWNLOAD_WEIGHTS = False",codes["parameters"])
        self.assertIn("ibm_full100_v2.zip",codes["parameters"])
        self.assertIn("planetary_vlm.colab_ibm_full",codes["commands"])
        self.assertIn("ibm-val14-colab-lock.txt",codes["install"])
        self.assertIn("summary['images'] == 100",codes["review"])
        self.assertIn("--resume",codes["run"])


if __name__=="__main__":
    unittest.main()

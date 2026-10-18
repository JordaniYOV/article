"""Reference and response-format fixtures; never scientific model findings."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from planetary_vlm.evaluation import evaluate
from planetary_vlm.evaluation.references import reference_status, READY, CANDIDATE, UNKNOWN


def fixture():
    requests,targets,predictions = [],[],[]
    for i,answer in enumerate(("YES","NO","NO")):
        rid = f"request{i}"
        requests.append(dict(request_id=rid,image_paths=["image.png"],prompt="Choose a label.",
            allowed_answers=["YES","NO"],max_new_tokens=8,do_sample=False))
        targets.append(dict(request_id=rid,answer=answer,source_sample_id=f"image{i}",group_id=f"scene{i}",
            derivation_version="synthetic-v1",task_id="crater_presence_v1"))
        predictions.append(dict(request_id=rid,raw_response="YES",status="ok",elapsed_seconds=.01))
    return requests,targets,predictions


class AdmissionTests(unittest.TestCase):
    def test_candidate_and_unknown_are_not_negative_ground_truth(self):
        req,gt,pred = fixture()
        gt[1].update(reference_status=CANDIDATE)
        gt[2].update(answer=None,reference_status=UNKNOWN)
        result = evaluate(req,gt,pred)
        self.assertEqual(result["overall"]["n_scored"],1)
        self.assertEqual(result["overall"]["n_unscored_reference"],2)
        self.assertEqual(result["overall"]["accuracy"],1)
        self.assertEqual(result["overall"]["reference_coverage"],1/3)
        self.assertIsNone(result["overall"]["binary"]["fpr"])
        for row in result["per_request"][1:]:
            self.assertIsNone(row["answer"])
            self.assertIsNone(row["correct"])
            self.assertEqual(row["scoring_status"],"unscored_reference")
        json.dumps(result,allow_nan=False)

    def test_all_unknown_returns_no_accuracy_not_zero(self):
        req,gt,pred = fixture()
        for row in gt:
            row.update(answer=None,reference_status=UNKNOWN)
        result = evaluate(req,gt,pred[:1])
        self.assertEqual(result["overall"]["n_failure"],0)
        self.assertEqual(result["overall"]["n_unscored_reference"],3)
        self.assertIsNone(result["overall"]["accuracy"])
        self.assertIsNone(result["overall"]["macro_f1"])
        self.assertEqual(result["per_request"][1]["response_status"],"missing")

    def test_ready_requires_real_reference_and_valid_enum(self):
        for status,answer in ((READY,None),("accepted","YES"),("UNKNOWN","NO")):
            records = fixture()
            records[1][0].update(reference_status=status,answer=answer)
            with self.subTest(status=status),self.assertRaises(ValueError):
                evaluate(*records)

    def test_missing_plain_answer_is_unknown_not_no(self):
        req,gt,pred = fixture()
        gt[0]["answer"] = None
        self.assertEqual(evaluate(req,gt,pred)["per_request"][0]["reference_status"],UNKNOWN)

    def test_reference_admission_flags_are_conservative(self):
        for row in ({"review_status":"pending_second_review"},{"scoring_eligible":False}):
            self.assertEqual(reference_status(row,has_reference=True)[0],CANDIDATE)
        self.assertEqual(reference_status({},has_reference=True,require_explicit=True)[0],CANDIDATE)
        self.assertEqual(reference_status({"review_status":"accepted"},has_reference=True,require_explicit=True)[0],READY)
        self.assertEqual(reference_status({"scored":False,"reference_status":READY},has_reference=True)[0],READY)
        for row in ({"review_status":"pending","reference_status":READY},
                    {"scoring_eligible":"false"},{"review_status":"unrecognized"}):
            with self.subTest(row=row),self.assertRaises(ValueError):
                reference_status(row,has_reference=True)

    def test_structured_contract_is_explicit_not_inferred_from_prose(self):
        req,gt,pred = fixture()
        pred[0]["raw_response"] = "ANSWER: YES\nEVIDENCE: A visible rim is in cell 5."
        pred[1]["raw_response"] = "ANSWER: NO"
        pred[2]["raw_response"] = "The answer is NO; ANSWER: NO"
        result = evaluate(req,gt,pred,response_format="label_evidence")
        self.assertEqual(result["overall"]["n_valid"],1)
        self.assertEqual(result["overall"]["n_invalid"],2)
        self.assertEqual(result["overall"]["n_correct"],1)
        self.assertEqual(result["per_request"][1]["recognized_label"],"NO")
        self.assertIsNone(result["per_request"][1]["parsed_answer"])
        self.assertFalse(result["per_request"][1]["correct"])
        self.assertEqual(sum(sum(r) for r in result["overall"]["confusion_matrix"]["counts"]),3)
        self.assertEqual(evaluate(req,gt,pred)["overall"]["n_valid"],0)
        with self.assertRaises(ValueError):
            evaluate(req,gt,pred,response_format="auto")

    def test_q1_q4_stay_separate_and_do_not_mutate_inputs(self):
        req,gt,pred = fixture()
        req[2]["allowed_answers"] = [str(i) for i in range(1,10)]+["NONE","TIE"]
        gt[2].update(answer="5",task_id="crater_largest_location_v1")
        pred[2]["raw_response"] = "5"
        before = copy.deepcopy((req,gt,pred))
        result = evaluate(req,gt,pred)
        self.assertIsNone(result["overall"]["macro_f1"])
        self.assertEqual(len(result["slices"]),2)
        self.assertEqual((req,gt,pred),before)


if __name__ == '__main__':
    unittest.main()

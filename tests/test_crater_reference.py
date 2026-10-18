"""Synthetic geometry only; no real images, downloads, inference or model outputs."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planetary_vlm.datasets.crater_reference import TASKS, candidate_answers, geometry, prompt_for


def crater(box=(10,10,30,30), **fields):
    return {"instance_id": "fixture", "bbox_xyxy": list(box), "uncertain": False, "truncated": False, **fields}


def answers(objects, size=(100,100), **kwargs):
    return [q["candidate_answer"] for q in candidate_answers(objects,*size,**kwargs)[0]]


class CraterReferenceTests(unittest.TestCase):
    def test_empty_requires_complete_reference(self):
        self.assertEqual(answers([]), ["NO","NO","NONE","TIE"])
        self.assertEqual(answers([],source_complete=False), [None]*4)
        self.assertEqual(answers([],image_complete=False), [None]*4)

    def test_one_center_and_right_midline(self):
        self.assertEqual(answers([crater((40,40,60,60))]), ["YES","YES","5","RIGHT"])

    def test_uneven_grid_uses_floor_boundaries_not_scaled_index(self):
        obj = crater((79.5,79.5,90.5,90.5))
        self.assertEqual(geometry(obj,256,256)["cell"],5)
        self.assertEqual(geometry(crater((164.5,164.5,175.5,175.5)),256,256)["cell"],9)

    def test_inclusive_size_and_exact_margin(self):
        self.assertTrue(geometry(crater((1,1,12,12)),256,256)["eligible"])
        for box in ((.99,1,12,12),(1,1,11.99,12),(244,244,255.01,255.01)):
            self.assertFalse(geometry(crater(box),256,256)["eligible"])

    def test_small_and_clipped_stay_out(self):
        self.assertEqual(answers([crater((10,10,13,13)),crater(truncated=True)]),["NO","NO","NONE","TIE"])
        self.assertEqual(answers([crater((-5,-5,20,20),truncated=True)]),["NO","NO","NONE","TIE"])

    def test_largest_tie_is_cross_cell_not_object_count(self):
        self.assertEqual(answers([crater((5,5,25,25)),crater((70,70,90,90))])[2],"TIE")
        self.assertEqual(answers([crater((5,5,25,25)),crater((6,6,26,26))])[2],"1")

    def test_largest_uses_sqrt_area_and_95_percent_rule(self):
        large=crater((5,5,25,25))
        self.assertEqual(answers([large,crater((70,70,89,89))])[2],"TIE")
        self.assertEqual(answers([large,crater((70,70,88.9,88.9))])[2],"1")

    def test_unknown_geometry_is_not_absence_or_small_based_on_approx_box(self):
        unknown=crater((1,1,2,2),uncertain=True,truncated=True)
        self.assertEqual(answers([unknown]),[None]*4)
        self.assertEqual(answers([unknown,crater((40,40,60,60))]),["YES","YES",None,None])

    def test_fixed_class_only_requires_explicit_geometry_and_single_instance_review(self):
        unknown=crater((70,70,90,90),uncertain=True,uncertainty_scope="class_only",geometry_reviewed=True)
        self.assertEqual(answers([unknown]),[None]*4)
        unknown["single_instance_confirmed"]=True
        self.assertEqual(answers([unknown]),[None,"NO",None,None])

    def test_question_specific_count_invariance(self):
        possible=crater((75,70,85,80),uncertain=True,uncertainty_scope="class_only",geometry_reviewed=True,single_instance_confirmed=True)
        certain=[crater((5+i*5,5,15+i*5,15)) for i in range(4)]+[crater((70,5,80,15))]
        self.assertEqual(answers(certain+[possible])[3],"LEFT")
        certain=certain[:3]+[crater((70,5,80,15)),crater((70,25,80,35))]
        self.assertIsNone(answers(certain+[possible])[3])

    def test_unknown_regions_and_duplicate_identity_keep_only_existential_certificates(self):
        for flag in ({"unknown_regions":True},{"duplicate_review_pending":True},{"source_complete":False}):
            self.assertEqual(answers([crater((40,40,60,60))],**flag),["YES","YES",None,None])

    def test_ambiguous_clipped_object_cannot_itself_certify_existence(self):
        obj=crater((40,40,60,60),reference_uncertainty='possible rim clipping')
        self.assertEqual(answers([obj]),[None]*4)
        self.assertEqual(answers([obj,crater((5,5,25,25))]),['YES',None,None,None])

    def test_label_inside_missing_visual_region_cannot_certify_existence(self):
        region=[{'bbox_xyxy':[35,35,65,65],'reason':'unresolved missing pixels'}]
        self.assertEqual(answers([crater((40,40,60,60))],unknown_regions=region),[None]*4)
        self.assertEqual(answers([crater((5,5,25,25))],unknown_regions=region),['YES',None,None,None])

    def test_class_only_confirmed_ineligible_does_not_poison_all_questions(self):
        obj=crater((10,10,11,11),uncertain=True,uncertainty_scope="class_only",geometry_reviewed=True,single_instance_confirmed=True)
        self.assertEqual(answers([obj]),["NO","NO","NONE","TIE"])

    def test_unreviewed_outputs_never_claim_scoring_admission(self):
        for row in candidate_answers([crater()],100,100)[0]:
            self.assertFalse(row["scored"])
            self.assertIsNone(row["reference_answer"])
            self.assertNotIn("UNKNOWN_REFERENCE",row["allowed_answers"])

    def test_prompts_have_all_rules_but_no_targets_or_annotation_content(self):
        for task in TASKS:
            prompt=prompt_for(task,256,256)
            self.assertIn("11 pixels",prompt)
            self.assertIn("85,170",prompt)
            self.assertIn("1 pixel margin",prompt)
            for forbidden in ("mask_sha256","source_catalog","candidate_answer","UNKNOWN_REFERENCE"):
                self.assertNotIn(forbidden,prompt)

    def test_invalid_shapes_and_types_fail_closed(self):
        for obj in (crater((1,1,1,3)),crater((0,0,float('nan'),3)),crater(uncertain="false"),crater((-1,0,20,20))):
            with self.subTest(obj=obj),self.assertRaises(ValueError): geometry(obj,100,100)


if __name__ == "__main__": unittest.main()

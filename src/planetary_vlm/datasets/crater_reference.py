"""Pure-stdlib Q1–Q4 geometry and conservative question-level uncertainty.

These are candidate references, not permission to score unreviewed annotations.
No semantic connected components, model outputs or hidden-rim inference.
"""
from itertools import product
import math

TASKS = ("crater_presence_v1", "crater_center_v1", "crater_largest_location_v1", "crater_half_count_v1")
ANSWERS = (('YES', 'NO'), ('YES', 'NO'), tuple(map(str, range(1, 10))) + ('NONE', 'TIE'), ('LEFT', 'RIGHT', 'TIE'))
VERSION = "crater-reference-v1"


def geometry(obj, width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < 3:
        raise ValueError("Invalid native dimensions")
    box = obj.get("bbox_xyxy")
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise ValueError("Expected bbox_xyxy")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in box):
        raise ValueError("Nonfinite/invalid bbox coordinate")
    x1, y1, x2, y2 = box
    if not x1 < x2 or not y1 < y2:
        raise ValueError("Nonpositive bbox extent")
    for name in ("uncertain", "truncated"):
        if type(obj.get(name)) is not bool:
            raise ValueError("Explicit boolean attributes required")
    if not obj["truncated"] and not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Out-of-frame geometry must retain truncated=true")
    cx, cy = (x1+x2)/2, (y1+y2)/2
    threshold = math.ceil(.04 * min(width, height))
    eligible = (not obj["truncated"] and min(x2-x1, y2-y1) >= threshold
                and x1 >= 1 and y1 >= 1 and x2 <= width-1 and y2 <= height-1)
    cell = 1 + sum(cx >= k*width//3 for k in (1, 2)) + 3*sum(cy >= k*height//3 for k in (1, 2))
    boundary = (abs(min(x2-x1, y2-y1)-threshold) <= 2 or abs(cx-width/2) <= 2
                or any(abs(cx-k*width//3) <= 2 or abs(cy-k*height//3) <= 2 for k in (1, 2)))
    return {"eligible": eligible, "center_xy": [cx, cy], "cell": cell,
            "half": "LEFT" if cx < width/2 else "RIGHT", "size_proxy": math.sqrt((x2-x1)*(y2-y1)),
            "threshold_px": threshold, "boundary_review": boundary}


def definite_answers(objects):
    cells = {o["cell"] for o in objects}
    if objects:
        maximum = max(o["size_proxy"] for o in objects)
        near = {o["cell"] for o in objects if o["size_proxy"] >= .95*maximum}
        largest = str(next(iter(near))) if len(near) == 1 else "TIE"
    else:
        largest = "NONE"
    delta = sum(1 if o["half"] == "LEFT" else -1 for o in objects)
    return ("YES" if objects else "NO", "YES" if 5 in cells else "NO", largest,
            "LEFT" if delta > 0 else "RIGHT" if delta < 0 else "TIE")


def candidate_answers(objects, width, height, *, source_complete=True, unknown_regions=False,
                      duplicate_review_pending=False, image_complete=True):
    """Return invariant answers or UNKNOWN_REFERENCE; never silently drop uncertain.

    Class-only uncertainty can enumerate all presence/absence worlds only after
    explicit geometry and single-instance review. Otherwise an uncertain bbox is
    NOT a bound on the possible object. Unknown source instances/regions remain
    unbounded; only confirmed existential positives Q1/Q2 can survive.
    """
    confirmed, possible = [], []
    unbounded = not source_complete or bool(unknown_regions) or not image_complete or duplicate_review_pending
    reasons = []
    details = []
    if not source_complete: reasons.append("source_instances_incomplete")
    if unknown_regions: reasons.append("unknown_visual_regions")
    if not image_complete: reasons.append("image_review_incomplete")
    if duplicate_review_pending: reasons.append("duplicate_identity_review_pending")
    for obj in objects:
        g = geometry(obj, width, height)
        obscured = False
        if isinstance(unknown_regions, list):
            for region in unknown_regions:
                if not isinstance(region, dict) or "bbox_xyxy" not in region:
                    continue
                a,b,c,d = region["bbox_xyxy"]
                x1,y1,x2,y2 = obj["bbox_xyxy"]
                if min(c,x2)>max(a,x1) and min(d,y2)>max(b,y1):
                    obscured = True
        visibility_unresolved = obscured or bool(obj.get("reference_uncertainty"))
        details.append({"instance_id": obj.get("instance_id"), **g, "reference_visibility_unresolved": visibility_unresolved})
        if obj["uncertain"] or visibility_unresolved:
            if visibility_unresolved:
                unbounded = True
                reasons.append("existing_object_visibility_or_eligibility_unresolved")
                continue
            if obj.get("definitely_ineligible_reviewed") is True:
                continue
            class_only = (obj.get("uncertainty_scope") == "class_only"
                          and obj.get("geometry_reviewed") is True
                          and obj.get("single_instance_confirmed") is True)
            if class_only:
                if g["eligible"]: possible.append(g)
            else:
                unbounded = True
                reasons.append("uncertain_class_or_geometry_unbounded")
        elif g["eligible"]:
            confirmed.append(g)
    values = [set() for _ in TASKS]
    if not unbounded and len(possible) <= 16:
        for include in product((False, True), repeat=len(possible)):
            answers = definite_answers(confirmed + [g for g, take in zip(possible, include) if take])
            for target, answer in zip(values, answers): target.add(answer)
    else:
        if len(possible) > 16: reasons.append("bounded_world_enumeration_limit")
        if confirmed: values[0].add("YES")
        if any(g["cell"] == 5 for g in confirmed): values[1].add("YES")
    result = []
    for task, labels, variants in zip(TASKS, ANSWERS, values):
        answer = next(iter(variants)) if len(variants) == 1 else None
        result.append({"task_id": task, "allowed_answers": list(labels),
                       "reference_status": "CANDIDATE_REFERENCE" if answer is not None else "UNKNOWN_REFERENCE",
                       "reference_answer": None, "candidate_answer": answer,
                       "admission_reason": "answer_invariant_pending_reference_review" if answer is not None else
                           ";".join(sorted(set(reasons))) or "uncertainty_changes_answer",
                       "scored": False})
    return result, details


def prompt_for(task_id, width, height):
    index = TASKS.index(task_id)
    prefix = (f"The image has native dimensions {width} by {height} pixels. Count impact craters, not pits, rocks, cones, or mare patches. "
              f"Use a tight axis-aligned bounding box around each visible outer rim. Only count fully visible craters with "
              f"a bbox short side at least {math.ceil(.04*min(width,height))} pixels and a 1 pixel margin from every image edge. "
              "Count separately identifiable nested or overlapping craters separately. The center is the bbox midpoint. "
              "Divide the image into a 3 by 3 grid numbered 1 2 3 / 4 5 6 / 7 8 9. "
              f"Column boundaries are x={width//3},{2*width//3}; row boundaries are y={height//3},{2*height//3}. "
              "A center on a grid boundary belongs to the cell to the right or below. ")
    questions = (
        "Is there at least one eligible crater? ",
        "Is there an eligible crater center in cell 5? ",
        "Which cell contains the center of the largest eligible crater? Size is sqrt(bbox width times height). "
        "Include every crater with size at least 95% of the maximum as a near-largest candidate. "
        "If all such centers are in one cell, return that cell; if they span multiple cells return TIE. If none exist return NONE. ",
        f"Which image half contains more eligible crater centers? LEFT means x<{width/2:g}; RIGHT means x>={width/2:g}. "
        "Return TIE for equal counts, including zero versus zero. ",
    )
    return prefix + questions[index] + "Reply with exactly one label: " + ", ".join(ANSWERS[index]) + ". No explanation."

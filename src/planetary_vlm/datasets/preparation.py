"""Shared clean semantic-track assembly with separate requests and targets.

No network, models, training or pseudo-labels. Requires the extracted official test
split supplied by the dataset class. Destination is immutable/exclusive.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path

from planetary_vlm.datasets.semantic import convert_semantic
from planetary_vlm.io import read_jsonl, write_json, write_jsonl


def build(source, specification, destination, *, domain="mars", depth_index=None):
    if domain not in {"mars", "moon"}:
        raise ValueError("Dataset domain must be mars or moon")
    source, specification, destination = map(Path, (source, specification, destination))
    if destination.exists():
        raise FileExistsError(f"Version already exists; choose a new destination: {destination}")
    rows = read_jsonl(source)
    if not rows or any(row.get("split") != "test" for row in rows):
        raise ValueError("This build requires the original official test split only")
    depths, depth_exclusions = {}, []
    if depth_index is not None:
        import numpy as np
        import tomllib
        from PIL import Image
        roi = tomllib.loads(specification.read_text(encoding='utf-8-sig'))['roi']
        entries = read_jsonl(depth_index)
        depths = {r['sample_id']:r for r in entries}
        if len(depths) != len(entries):
            raise ValueError('Duplicate depth reference sample')
        selected = []
        for sample in rows:
            entry = depths.get(sample['sample_id'])
            if entry is None or entry.get('status') != 'technical_depth_qa_pass':
                depth_exclusions.append({'sample_id':sample['sample_id'],'reason':'missing_QA_depth'})
                continue
            image_path = (source.parent/sample['image_path']).resolve()
            if entry['image_path'] != str(image_path) or hashlib.sha256(image_path.read_bytes()).hexdigest() != entry['registration']['benchmark_sha256']:
                raise ValueError('Depth refers to different RGB image')
            for field in ('depth','validity'):
                if hashlib.sha256(Path(entry[field+'_path']).read_bytes()).hexdigest() != entry[field+'_sha256']:
                    raise ValueError('Depth asset changed since QA')
            depth = np.load(entry['depth_path'],allow_pickle=False)
            valid = np.load(entry['validity_path'],allow_pickle=False).astype(bool)
            with Image.open(image_path) as image:
                if depth.shape != (image.height,image.width) or valid.shape != depth.shape:
                    raise ValueError('RGB/depth grid mismatch')
                box = [int(roi[i]*(image.width if i%2==0 else image.height)) for i in range(4)]
            valid &= np.isfinite(depth) & (depth>0)
            x0,y0,x1,y1 = box
            coverage = float(valid[y0:y1,x0:x1].mean())
            if coverage < .5:
                depth_exclusions.append({'sample_id':sample['sample_id'],'reason':'ROI_depth_coverage_below_50_percent','coverage':coverage})
                continue
            selected.append(dict(sample,image_path=str(image_path),mask_path=str((source.parent/sample['mask_path']).resolve())))
        rows = selected
        if not rows:
            raise ValueError('No depth-complete samples pass benchmark ROI coverage')
    destination.mkdir(parents=True, exist_ok=False)
    converter_source = source
    if depth_index is not None:
        converter_source = destination/'source_samples.jsonl'
        write_jsonl(converter_source, rows)
    track = convert_semantic(converter_source, specification, destination / domain / "test")
    requests = read_jsonl(track / "requests.jsonl")
    targets = read_jsonl(track / "targets.jsonl")
    if len(requests) != len(targets) or len({row["source_sample_id"] for row in targets}) != len(targets):
        raise ValueError("Expected exactly one question per admitted photograph")
    prohibited = {"answer", "mask_path", "prepared_mask", "ground_truth"}
    if any(prohibited & set(row) for row in requests):
        raise ValueError("Evaluation-only fields leaked into inference requests")
    if not requests:
        raise ValueError("No admissible questions")
    fingerprint = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0", "dataset_version": destination.name,
        "status": "partial_mars_clean_ready_moon_incomplete",
        "source_index_sha256": fingerprint(source),
        "specification_sha256": fingerprint(specification),
        "builder_sha256": fingerprint(Path(__file__)),
        "converter_sha256": fingerprint(Path(__file__).parent / "semantic.py"),
        "mars": {"source_photographs": len(rows), "admitted_photographs": len(targets),
                 "questions": len(requests), "split": "official_test_unchanged",
                 "task_counts": dict(Counter(row["task_id"] for row in targets)),
                 "answer_counts": dict(Counter(row["answer"] for row in targets)),
                 "groups": len({row["group_id"] for row in targets}),
                 "requests": "mars/test/requests.jsonl", "targets": "mars/test/targets.jsonl",
                 "provenance": "mars/test/provenance.jsonl", "depth_maps": 0,
                 "source": "https://zenodo.org/records/15494736",
                 "license": "CC-BY-4.0 (publisher claim; upstream audit pending)",
                 "training_overlap_audit": "pending", "human_question_qa": "pending"},
        "moon": {"requested_photographs": 300, "admitted_complete_triplets": 0,
                 "status": "blocked_missing_same_frame_semantic_masks_and_additional_stereo_data",
                 "depth_policy": "provided_or_stereo_only_no_monocular_predictions"},
        "conditions": ["clean"], "degradation_variants": "not_built",
        "unsupported_tasks": ["largest_rock_instance_location", "observed_traversability",
                              "preferred_direction_geometry_proxy"],
        "segmentation_masks": "evaluation_only_never_in_model_request",
        "scientific_results": False,
    }
    if domain == "moon":
        stats = manifest.pop("mars")
        stats.update(requests="moon/test/requests.jsonl", targets="moon/test/targets.jsonl",
                     provenance="moon/test/provenance.jsonl",
                     source=sorted({row["source_url"] for row in rows}),
                     license=sorted({row["license"] for row in rows}))
        manifest["moon"].update(stats, status="clean_semantic_only_depth_not_prepared")
        manifest["status"] = "moon_clean_semantic_only_mars_not_included"
    if depth_index is not None:
        admitted = {row['source_sample_id'] for row in targets}
        prepared_depths = []
        for provenance in read_jsonl(track/'provenance.jsonl'):
            if provenance['sample_id'] not in admitted:
                continue
            entry = depths[provenance['sample_id']]
            x0,y0,x1,y1 = provenance['roi_pixels']
            depth = np.load(entry['depth_path'],allow_pickle=False)[y0:y1,x0:x1]
            valid = np.load(entry['validity_path'],allow_pickle=False)[y0:y1,x0:x1].astype(bool)
            folder = track/'depth'
            folder.mkdir(exist_ok=True)
            token = Path(provenance['prepared_image']).stem
            dp,vp = folder/(token+'.range_m.npy'), folder/(token+'.validity.npy')
            np.save(dp, np.where(valid,depth,np.nan),allow_pickle=False)
            np.save(vp, valid,allow_pickle=False)
            prepared_depths.append({**entry,'depth_path':str(dp),'validity_path':str(vp),
                'depth_sha256':fingerprint(dp),'validity_sha256':fingerprint(vp),
                'image_path':provenance['prepared_image'],'mask_path':provenance['prepared_mask'],
                'full_frame_depth_path':entry['depth_path'],'valid_fraction':float(valid.mean()),
                'roi_pixels':provenance['roi_pixels']})
        write_jsonl(track/'depth_index.jsonl',prepared_depths)
        write_jsonl(destination/'depth_exclusions.jsonl',depth_exclusions)
        manifest[domain].update(depth_maps=len(prepared_depths),depth_index=f'{domain}/test/depth_index.jsonl',
            depth_provenance='stereo_reconstructed_not_independent_depth_GT',every_admitted_photo_has_depth=True)
        manifest.update(status=f'{domain}_clean_stereo_depth_subset_pending_scientific_review',
            depth_index_sha256=fingerprint(Path(depth_index)),ROI_depth_min_valid_fraction=.5)
    write_json(destination / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return destination

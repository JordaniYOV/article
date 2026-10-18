"""Frozen 100-image IBM clean run; common detector, smoke gate and checkpoints.

No tuning, corruption, new reference admission or implicit asset acquisition.
The separate evaluate operation reads only the local, non-inference eval kit.
"""
import argparse
from collections import Counter
import json

from planetary_vlm import colab_ibm as ibm
from planetary_vlm.colab import verify_bundle


def cohort(root):
    manifest = verify_bundle(root)
    images = ibm.select_inputs(root)
    if (manifest.get("selected_images") != 100 or manifest.get("requests") != 400
        or manifest["protocol"].get("conditions") != ["clean"]
        or manifest["protocol"].get("inference_cohort") is not None
        or set(manifest["protocol"].get("source_ids", [])) != {r["sample_id"] for r in images}
        or len(images) != 100 or Counter(r["planet"] for r in images) != {"moon":50,"mars":50}):
        raise ValueError("Full IBM run requires the frozen 100 clean images / 400 Q1-Q4 requests")
    ibm.select_inputs(root, smoke=True)
    return manifest, images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify","environment","assets","smoke","run","evaluate"))
    parser.add_argument("--workspace")
    parser.add_argument("--persist")
    parser.add_argument("--run-id", default="ibm_full100_v2_clean")
    parser.add_argument("--session-id")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--results")
    parser.add_argument("--reference-dir")
    parser.add_argument("--output")
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        if not all((args.results,args.reference_dir,args.output)):
            parser.error("evaluate requires --results, --reference-dir and --output")
        from planetary_vlm.colab_ibm_pilot import score_results
        result = score_results(args.results,args.reference_dir,args.output,
            provisional=args.provisional,scope="full100")
    else:
        if not args.workspace:
            parser.error("Inference operations need --workspace")
        manifest, images = cohort(args.workspace)
        if args.command == "verify":
            from PIL import Image
            formats = Counter()
            for row in images:
                with Image.open(row["image_path"]) as image:
                    if image.format not in {"PNG","TIFF"} or image.mode not in {"L","RGB"}:
                        raise ValueError("Expected original 8-bit PNG/TIFF inputs")
                    image.load()
                    formats[image.format] += 1
            result = {"images":100,"answers":400,"counts":manifest["counts"],
                "actual_source_formats":dict(formats),"bundle_sha256":manifest["bundle_sha256"],
                "scientific_score_available":False,
                "limitations":"Includes prior development scenes; source val/test labels are retained, not a new untouched test set."}
        elif args.command == "assets":
            result = ibm.acquire_assets(args.workspace,download=args.download)
        elif args.command == "environment":
            result = ibm.environment(args.workspace)
        else:
            if not args.persist or not args.session_id:
                parser.error("smoke/run requires --persist and --session-id")
            result = ibm.run_job(args.workspace,args.persist,args.run_id,args.session_id,
                smoke=args.command=="smoke",resume=args.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

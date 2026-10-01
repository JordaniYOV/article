"""Paired, scene-group uncertainty for two completed frozen-encoder probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from linear_head_approach.linear_probe import paired_group_difference

ROOT = Path(__file__).resolve().parents[1]


def run(first: Path, second: Path, input_manifest: Path) -> dict:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (first, second)]
    inputs = [json.loads(line) for line in input_manifest.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    groups = {row["sample_id"]: row["scene_group_id"] for row in inputs if row["split"] == "test"}
    if len(groups) != 31 or any(set(report["per_image_scores"]) != set(groups) for report in reports):
        raise ValueError("Reports must cover the same 31 publisher-test samples")
    if reports[0]["extractor_id"] == reports[1]["extractor_id"]:
        raise ValueError("Expected different encoders")
    if any(report.get("test_images") != 31 for report in reports):
        raise ValueError("Unexpected test report size")
    result = paired_group_difference(reports[0]["per_image_scores"],
                                     reports[1]["per_image_scores"], groups)
    result.update(first_encoder=reports[0]["extractor_id"],
                  second_encoder=reports[1]["extractor_id"],
                  first_report=str(first), second_report=str(second),
                  article_level_claim=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-report", type=Path, required=True)
    parser.add_argument("--second-report", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, default=ROOT / "linear_head_approach" / "outputs" / "encoder_probe_wac_v1" / "encoder_inputs.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.first_report, args.second_report, args.input_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

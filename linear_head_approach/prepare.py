"""Build the WAC crater encoder-probe manifests from the curated orbital set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from linear_head_approach.encoder_probe import write_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    output = arguments.output or root / "linear_head_approach" / "outputs" / "encoder_probe_wac_v1"
    print(json.dumps(write_probe(root, output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

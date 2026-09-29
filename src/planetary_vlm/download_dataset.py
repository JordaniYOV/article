"""Explicit acquisition entrypoint; importing this module does not use network."""
import argparse
import json
from planetary_vlm.datasets import MarsData, LuneData


def download_dataset(*, planet="all", root="data_rover", mars_archive=False,
                     lunar_pilot_ids=("001", "084", "168"), stereo_stage=None,
                     images=False):
    if planet not in {"mars", "moon", "all"}:
        raise ValueError("Select mars, moon or all")
    if stereo_stage is not None:
        if images and stereo_stage != "pilot":
            raise ValueError("--images is only applicable to --stereo-stage pilot")
        if planet != "mars" or mars_archive:
            raise ValueError("Stereo acquisition requires --planet mars and no --mars-archive")
        return MarsData(root).download_stereo_sources(stereo_stage, images=images)
    if images:
        raise ValueError("--images requires --stereo-stage pilot")
    if mars_archive and planet == "moon":
        raise ValueError("--mars-archive cannot be used with --planet moon")
    results = {}
    if planet in {"moon", "all"}:
        results["moon"] = LuneData(root).download(pilot_ids=lunar_pilot_ids)
    if planet in {"mars", "all"}:
        results["mars"] = MarsData(root).download(archive=mars_archive)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", choices=("mars", "moon", "all"), default="all")
    parser.add_argument("--root", default="data_rover")
    parser.add_argument("--mars-archive", action="store_true")
    parser.add_argument("--lunar-pilot-ids", nargs="*", default=["001", "084", "168"])
    parser.add_argument("--stereo-stage", choices=("catalog", "labels", "aliases", "images", "pilot"))
    parser.add_argument("--images", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(download_dataset(**vars(args)), default=str, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Offline dataset assembly; does not download sources or query models."""
import argparse
from pathlib import Path
from planetary_vlm.datasets import MarsData, LuneData


def prepare_dataset(*, planet, output, root="data_rover", samples=None, specification=None,
                    depth_index=None, track=None):
    classes = {"mars": MarsData, "moon": LuneData}
    if planet not in classes:
        raise ValueError("Select mars or moon")
    options = dict(output=output, samples=samples, specification=specification)
    if depth_index is not None:
        options['depth_index'] = depth_index
    if track is not None:
        if planet != "moon" or track != "lusnar":
            raise ValueError("The only explicit lunar preparation track is 'lusnar'")
        options['track_metadata'] = {
            "track": "synthetic_lunar_rover",
            "source": "https://huggingface.co/datasets/xuboluo2001/LuSNAR",
            "license": "MIT (publisher claim; verify current terms before redistribution)",
            "synthetic": True,
            "scene_split": "Publisher scene split retained per sample; custom balanced selection",
            "scientific_scope": "Synthetic sim-to-real diagnostic only; not evidence on real lunar imagery",
            "depth_provenance": "Unreal Engine simulator ground truth; native PFM values, metric units unverified",
        }
    result = classes[planet](root).prepare(**options)
    if track == "lusnar" and depth_index is not None:
        from planetary_vlm.datasets.lusnar import make_lusnar_track_portable
        validation = None
        if samples is not None:
            candidate = Path(samples).resolve().parent / "source_validation.json"
            validation = candidate if candidate.is_file() else None
        result = make_lusnar_track_portable(result, source_validation=validation)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", choices=("mars", "moon"), required=True)
    parser.add_argument("--root", default="data_rover")
    parser.add_argument("--samples")
    parser.add_argument("--specification")
    parser.add_argument('--depth-index')
    parser.add_argument("--track", choices=("lusnar",))
    parser.add_argument("--output", required=True)
    print(prepare_dataset(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()

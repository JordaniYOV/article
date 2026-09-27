"""Offline dataset assembly; does not download sources or query models."""
import argparse
from planetary_vlm.datasets import MarsData, LuneData


def prepare_dataset(*, planet, output, root="data", samples=None, specification=None, depth_index=None):
    classes = {"mars": MarsData, "moon": LuneData}
    if planet not in classes:
        raise ValueError("Select mars or moon")
    options = dict(output=output, samples=samples, specification=specification)
    if depth_index is not None:
        if planet != 'mars':
            raise ValueError('Stereo batch depth index currently supports Mars only')
        options['depth_index'] = depth_index
    return classes[planet](root).prepare(**options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planet", choices=("mars", "moon"), required=True)
    parser.add_argument("--root", default="data")
    parser.add_argument("--samples")
    parser.add_argument("--specification")
    parser.add_argument('--depth-index')
    parser.add_argument("--output", required=True)
    print(prepare_dataset(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()

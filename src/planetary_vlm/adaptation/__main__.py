"""python -m planetary_vlm.adaptation <approach> <stage> ..."""
import argparse
import sys
from .common.config import METHODS


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("method", choices=[*METHODS, *(name.replace("_", "-") for name in METHODS)])
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args, rest = parser.parse_known_args(argv)
    from .common.cli import main as run
    return run(args.method.replace("-", "_"), rest)


if __name__ == "__main__":
    raise SystemExit(main())

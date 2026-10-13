"""One SpaceLLaVA job, in a fresh process using the service's interpreter."""

from pathlib import Path
import sys

if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from benchmark_services.common.worker import main
    main("spacellava")

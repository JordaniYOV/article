"""Selected-job CLI shared by model-specific worker entry points."""

import argparse


def main(service, argv=None):
    parser = argparse.ArgumentParser(description=f"Run one {service} job in this interpreter")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mock", action="store_true", help="Explicit non-scientific CPU fixture")
    args = parser.parse_args(argv)
    # Argument parsing/help never imports Torch or opens the queue.
    from .execution import run_selected
    from .protocol import BACKENDS
    from .settings import Settings
    run_selected(Settings(), args.run_id, BACKENDS[service], mock_mode=args.mock)

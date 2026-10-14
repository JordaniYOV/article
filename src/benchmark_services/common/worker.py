"""Selected-job CLI shared by model-specific worker entry points."""

import argparse
import os
import time


def main(service, argv=None):
    parser = argparse.ArgumentParser(description=f"Run one {service} job in this interpreter")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mock", action="store_true", help="Explicit non-scientific CPU fixture")
    args = parser.parse_args(argv)
    # Argument parsing/help never imports Torch or opens the queue.
    from .execution import run_selected
    from .protocol import BACKENDS
    from .settings import Settings
    from api.logging import get_logger
    logger = get_logger("worker.cli", service=service, run_id=args.run_id, mock=args.mock,
        attempt_id=os.environ.get("PLANETARY_SERVICE_ATTEMPT_ID"))
    started = time.perf_counter()
    logger.info("worker_started")
    try:
        claimed = run_selected(Settings(), args.run_id, BACKENDS[service], mock_mode=args.mock)
    except Exception as error:
        logger.error("worker_failed", error_type=type(error).__name__, exit_code=1,
            elapsed_seconds=time.perf_counter() - started)
        # No raw traceback: dependency errors can contain options, paths or secrets.
        raise SystemExit(1) from None
    logger.info("worker_exited", claimed=claimed, exit_code=0, elapsed_seconds=time.perf_counter() - started)

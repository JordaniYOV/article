"""Independent B pipeline with the shared bounded frozen-feature cache."""
def main(argv=None):
    from ..common.cli import main as run
    return run("frozen_decoder", argv)

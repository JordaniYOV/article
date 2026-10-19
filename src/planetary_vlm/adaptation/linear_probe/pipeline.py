"""Independent A pipeline; orchestration/IO are shared, not copied."""
def main(argv=None):
    from ..common.cli import main as run
    return run("linear_probe", argv)

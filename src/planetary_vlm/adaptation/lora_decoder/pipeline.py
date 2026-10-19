"""Independent C pipeline; online encoding, frozen base, LoRA-only updates."""
def main(argv=None):
    from ..common.cli import main as run
    return run("lora_decoder", argv)

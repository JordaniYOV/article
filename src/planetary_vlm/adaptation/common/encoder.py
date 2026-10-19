"""Pinned publisher LunarBackbone, real NAC input, explicit LoRA and checkpointing."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .io import sha, within

RUNTIME_REVISION = "d54c67aad513cb9daca444afa425cfb278e4fbf8"
ASSET_HASHES = {"checkpoint.pt": "5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b",
                "config.yaml": "fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619"}


def inspect_assets(root, cfg):
    if sys.version_info[:2] not in ((3, 11), (3, 12)):
        raise RuntimeError("Pinned IBM runtime supports Python 3.11/3.12; use Python 3.12")
    root = Path(root).resolve()
    runtime = within(root, cfg["encoder"]["runtime"])
    if not runtime.is_relative_to(root / "external") or not runtime.is_dir():
        raise ValueError("Pinned IBM source is required under external/; no implicit clone")
    revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "--", "ni_lfm", "terratorch_integration"], text=True)
    if revision != RUNTIME_REVISION or dirty.strip():
        raise ValueError("IBM source revision differs or is dirty; it is never reset automatically")
    assets = within(root, cfg["encoder"]["assets"])
    if not assets.is_relative_to(root / "weights"):
        raise ValueError("Use explicit local weights/, never dataset assets")
    hashes = {}
    for name, expected in ASSET_HASHES.items():
        path = within(assets, name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Missing/changed pinned foundation asset: {path}")
        hashes[name] = expected
    source_hashes = {p.relative_to(runtime).as_posix(): sha(p) for folder in ("ni_lfm", "terratorch_integration")
                     for p in sorted((runtime / folder).rglob("*.py"))}
    return {"runtime_revision": revision, "runtime_files": source_hashes, "foundation_weights": hashes,
            "source_model_revision": "b657bdfe6fc4c0445d918f77a2d7b919805f3bce"}


class LowRankLinear(nn.Module):
    """Frozen fused-QKV projection plus alpha/r * B(A(x)); B starts at zero."""
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("LoRA target must be a linear projection")
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=5 ** .5)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        from torch.nn import functional as F
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scale


class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, mask=None):
        if torch.is_grad_enabled():
            return checkpoint(self.block, x, mask=mask, use_reentrant=False)
        return self.block(x, mask=mask)


class Encoder(nn.Module):
    def __init__(self, backbone, cfg):
        super().__init__()
        self.backbone, self.cfg = backbone, cfg
        self.adapters = []
        self.requires_grad_(False)
        self.eval()

    def add_lora(self):
        if self.adapters:
            raise ValueError("LoRA was already injected")
        l = self.cfg["lora"]
        blocks = self.backbone.model.encoder
        if len(blocks) != 12:
            raise ValueError("Expected 12 ViT-B encoder blocks")
        for block in blocks:
            adapter = LowRankLinear(block.attn.qkv, l["rank"], l["alpha"], l["dropout"])
            block.attn.qkv = adapter
            self.adapters.append(adapter)
        if l["checkpoint_blocks"]:
            self.backbone.model.encoder = nn.ModuleList([CheckpointBlock(block) for block in blocks])

    def train(self, mode=True):
        # Frozen dropout/stochastic depth never becomes active. Only adapter dropout changes.
        super().train(False)
        for adapter in self.adapters:
            adapter.dropout.train(mode)
        return self

    def forward(self, x):
        e = self.cfg["encoder"]
        if x.ndim != 4 or x.shape[1:] != (1, e["input_size"], e["input_size"]):
            raise ValueError("Expected declared single-intensity input, no replicated VIS5")
        hidden = self.backbone({"nac": x})
        side = e["input_size"] // e["patch_size"]
        outputs = []
        for layer in e["taps"]:
            tokens = hidden[layer]["nac"]
            if tokens.shape[1:] != (side * side, e["channels"]):
                raise ValueError("Unexpected native NAC feature grid")
            outputs.append(tokens.transpose(1, 2).reshape(len(x), e["channels"], side, side))
        return outputs


def load(root, cfg, device):
    verified = inspect_assets(root, cfg)
    runtime = within(root, cfg["encoder"]["runtime"])
    assets = within(root, cfg["encoder"]["assets"])
    if "ni_lfm" in sys.modules:
        loaded = Path(sys.modules["ni_lfm"].__file__).resolve()
        if not loaded.is_relative_to(runtime):
            raise ValueError("Another ni_lfm installation was imported; use a fresh process")
    sys.path.insert(0, str(runtime))
    spec = importlib.util.spec_from_file_location("adaptation_pinned_backbone", runtime / "terratorch_integration/lunar_backbone.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Checked(module.LunarBackbone):
        def load_checkpoint(self, path):
            state = torch.load(path, map_location="cpu", weights_only=True)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            incompatible = self.model.load_state_dict(state, strict=False)
            # Other modalities may be unexpected; no active parameter/buffer may be missing.
            if incompatible.missing_keys:
                raise ValueError(f"Foundation NAC weights/buffers would be uninitialized: {incompatible.missing_keys[:8]}")

    backbone = Checked(variant="base", modalities=["nac"], merge_method="dict",
                       cfg=str(assets / "config.yaml"), checkpoint_path=str(assets / "checkpoint.pt"),
                       patch_size=cfg["encoder"]["patch_size"])
    info = backbone.model.modality_info["nac"]
    if info["num_channels"] != 1:
        raise ValueError("Publisher NAC configuration is not single-channel")
    # Preserve the publisher's full forward. Unused frozen generative decoder stays frozen.
    return Encoder(backbone, cfg).to(device), verified


def trainable_state(module):
    return {name: value.detach().cpu().clone() for name, value in module.named_parameters() if value.requires_grad}


def restore_trainable(module, state):
    parameters = dict(module.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if set(state) != expected:
        raise ValueError("Trainable checkpoint keys differ from declared adaptation")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape:
                raise ValueError("Trainable checkpoint shape differs")
            parameters[name].copy_(value)

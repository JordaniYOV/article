"""One explicit protocol, independently configurable approach entrypoints."""
from copy import deepcopy
from pathlib import Path
import tomllib

METHODS = ("linear_probe", "frozen_decoder", "lora_decoder", "sam_baseline")
DEFAULTS = {
    "schema": "mars-adaptation-v1", "seeds": [13, 37, 73],
    "dataset": "data_orbital/mars_crater_binary_v1",
    "counts": {"train": 3600, "val": 900, "test": 900},
    "encoder": {"runtime": "external/ibm-lfm", "assets": "weights/encoder_heads/ibm/backbone",
                "input_size": 256, "patch_size": 16, "taps": [2, 5, 8, 11],
                "modality": "nac", "channels": 768, "precision": "fp16"},
    "training": {"max_epochs": 40, "patience": 8, "batch_size": 1, "accumulate": 4,
                 "head_lr": 0.0003, "lora_lr": 0.0001, "weight_decay": 0.01,
                 "dice_weight": 0.5, "positive_weight_cap": 10., "gradient_clip": 1.,
                 "thresholds": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
    "decoder": {"width": 64}, "lora": {"rank": 4, "alpha": 8., "dropout": 0., "checkpoint_blocks": True},
    "sam": {"checkpoint": "weights/sam/sam_vit_b_01ec64.pth", "sha256": "",
            "points_per_side": 32, "points_per_batch": 16, "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95, "min_area": 16, "max_area_fraction": 0.35,
            "min_axis_ratio": 0.3, "min_circularity": 0.35},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"Unknown protocol option: {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected table: {key}")
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load(method, path=None):
    method = method.replace("-", "_")
    if method not in METHODS:
        raise ValueError("Unknown adaptation approach")
    default = Path(__file__).parents[1] / method / "config.toml"
    with (Path(path) if path else default).open("rb") as stream:
        values = tomllib.load(stream)
    declared = values.pop("method")
    if declared != method:
        raise ValueError("Config belongs to another approach")
    cfg = merge(DEFAULTS, values)
    cfg["method"] = method
    validate(cfg)
    return cfg


def validate(cfg):
    if cfg["schema"] != "mars-adaptation-v1" or cfg["method"] not in METHODS:
        raise ValueError("Unknown protocol")
    e, t, d, l, s = (cfg[k] for k in ("encoder", "training", "decoder", "lora", "sam"))
    if (e["modality"], e["channels"]) != ("nac", 768) or e["precision"] != "fp16":
        raise ValueError("Only declared NAC single-intensity protocol is implemented")
    if (e["input_size"] not in (128, 256, 512) or e["patch_size"] not in (8, 16)
            or e["input_size"] % e["patch_size"] or e["taps"] != [2, 5, 8, 11]):
        raise ValueError("Unsupported image/patch grid or depth taps")
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]) or any(
            not isinstance(seed, int) or seed < 0 for seed in cfg["seeds"]):
        raise ValueError("Declare distinct nonnegative training seeds")
    for key in ("max_epochs", "patience", "batch_size", "accumulate"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"Invalid {key}")
    if (not 0 < t["head_lr"] < 1 or not 0 < t["lora_lr"] < 1 or t["weight_decay"] < 0
            or t["dice_weight"] < 0 or t["positive_weight_cap"] < 1 or t["gradient_clip"] <= 0):
        raise ValueError("Invalid training hyperparameters")
    if (not t["thresholds"] or t["thresholds"] != sorted(set(t["thresholds"]))
            or not all(0 < x < 1 for x in t["thresholds"])):
        raise ValueError("Declare an increasing validation threshold grid")
    if d["width"] < 8 or d["width"] % 8 or not 1 <= l["rank"] <= 64 or l["alpha"] <= 0 or not 0 <= l["dropout"] < 1:
        raise ValueError("Invalid decoder/LoRA dimensions")
    if (min(s["points_per_side"], s["points_per_batch"], s["min_area"]) < 1
            or not 0 < s["max_area_fraction"] <= 1 or not 0 <= s["min_circularity"] <= 1
            or not 0 < s["min_axis_ratio"] <= 1):
        raise ValueError("Invalid SAM filter")

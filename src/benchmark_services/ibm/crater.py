"""Published NAC crater detector; image-only inference and fixed Q1–Q4 readout.

No prompt interpretation, reference annotations, retraining or LLM is involved.
PNG transfer outside the native NAC dataset is exploratory, not reproduction.
"""
from copy import deepcopy
import gc
from hashlib import sha256
import math
from pathlib import Path

from .adapter import IBMIMPAdapter
from ..common.local_runtime import hashes, import_local, offline_imports
from planetary_vlm.datasets.crater_reference import TASKS, definite_answers, geometry

CODE_REVISION = "d54c67aad513cb9daca444afa425cfb278e4fbf8"
HEAD_ID = "nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model"
HEAD_REVISION = "f09ebeae2099c28f09bf4060f56d3e327a6192de"
HEAD_FILE = "NAC_ni_lfm_ps8_s44.ckpt"
HEAD_SHA256 = "00ecf2973a275e67fa4f9e9e742e5c2bf98f2291905b26e65468d70544e56a41"
CONFIG_SHA256 = "72c69d7508660e928567ce95432431034ba436d376538f90c116906de6f6fc1e"
BACKBONE_ID = "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"
BACKBONE_REVISION = "b657bdfe6fc4c0445d918f77a2d7b919805f3bce"
BACKBONE_SHA256 = "5843824dfa56f89552b37b5295aa865ed28777c5f2b33286db2ef81b4cf08e5b"
BACKBONE_CONFIG_SHA256 = "fd94d20eb2b1cd4379516da3d2c1912a56a8684e1952ee9331abd63e7c59e619"
PROTOCOL = "ibm_nac_craters_png_gray256_dn255_v1"
SCORE_THRESHOLD = 0.5  # Publisher visualization threshold, fixed before inference.
MEAN, STD, SIDE = 0.346548, 0.159291, 256


def prepare_png(path):
    """Declare a transfer protocol, not invented multispectral channels/radiance."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        if image.mode not in {"L", "RGB"} or min(image.size) < 3:
            raise ValueError("Expected an 8-bit L/RGB planetary PNG")
        width, height = image.size
        gray = image.convert("L").resize((SIDE, SIDE), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = ((pixels - MEAN) / STD)[None, None].copy()
    return tensor, {"native_width": width, "native_height": height,
        "model_width": SIDE, "model_height": SIDE,
        "input_tensor_sha256": sha256(tensor.tobytes()).hexdigest(),
        "normalization": {"divide_by": 255, "mean": MEAN, "std": STD},
        "resize": "Pillow bilinear to 256x256; boxes rescaled to native coordinates",
        "input_protocol": PROTOCOL}


def readout(boxes, scores, labels, width, height):
    """Validate native-coordinate predictions, then reuse the frozen geometry."""
    if len(boxes) != len(scores) or len(boxes) != len(labels):
        raise ValueError("Detector output lengths differ")
    all_boxes, eligible = [], []
    for box, score, label in zip(boxes, scores, labels):
        if (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
            or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
            or type(label) is not int or label != 1):
            raise ValueError("Invalid crater detection")
        obj = {"bbox_xyxy": list(box), "uncertain": False, "truncated": False}
        info = geometry(obj, width, height)
        keep = score >= SCORE_THRESHOLD and info["eligible"]
        all_boxes.append({"bbox_xyxy": list(box), "score": score, "label": label,
                          "eligible": keep, "geometry": info})
        if keep:
            eligible.append(info)
    return {"detections": all_boxes, "eligible_count": len(eligible),
            "answers": dict(zip(TASKS, definite_answers(eligible))),
            "score_threshold": SCORE_THRESHOLD, "geometry_version": "crater-reference-v1"}


def validate_model_keys(missing, unexpected):
    # NAC config is FULL fine-tuning, not WAC LoRA/frozen IMP. An omitted
    # encoder must fail, not silently revert to the pretrained backbone.
    missing_model = [k for k in missing if k.startswith("model.")]
    unexpected_model = [k for k in unexpected if k.startswith("model.")]
    if missing_model or unexpected_model:
        raise ValueError(f"Incomplete NAC detector checkpoint: missing={missing_model[:8]}, "
                         f"unexpected={unexpected_model[:8]}")


def validate_config(config):
    if config.get("model", {}).get("class_path") != "terratorch_integration.LunarObjectDetectionTask":
        raise ValueError("Expected published LunarObjectDetectionTask")
    args = deepcopy(config["model"]["init_args"])
    model = args["model_args"]
    data = config["data"]["init_args"]
    expected = {"framework": "faster-rcnn", "backbone": "ni_lfm_v1_base",
                "backbone_modalities": ["nac"], "backbone_patch_size": 8, "num_classes": 2,
                "in_channels": 1, "framework_min_size": SIDE, "framework_max_size": SIDE}
    if (any(model.get(k) != v for k, v in expected.items()) or args.get("freeze_backbone") is not False
        or args.get("peft_config") is not None or data.get("nac_norm_mean") != MEAN
        or data.get("nac_norm_std") != STD or data.get("image_size") != SIDE):
        raise ValueError("NAC detector architecture/normalization differs from release")
    return args


class IBMCraterDetector(IBMIMPAdapter):
    """Reuses offline asset checks/lifetime only; never the IMP preprocessing."""
    def preflight(self):
        if self.options.get("input_protocol") != PROTOCOL:
            raise ValueError("Explicit NAC crater PNG transfer protocol is required")
        from ..common.local_runtime import checked_repo, local_path
        if self.options.get("code_revision") != CODE_REVISION or self.options.get("local_files_only") is not True:
            raise ValueError("Use the pinned offline IBM runtime")
        repo = checked_repo(self.options, "terratorch_integration")
        paths = {k: local_path(self.options, k) for k in
                 ("head_config_path", "head_checkpoint_path", "backbone_config_path", "backbone_checkpoint_path")}
        return repo, paths

    def initialize(self):
        repo, paths = self.preflight()
        import torch
        import yaml
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("IBM Colab pilot requires CUDA; no silent CPU fallback")
        torch.manual_seed(44)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config = yaml.safe_load(paths["head_config_path"].read_text(encoding="utf-8"))
        args = validate_config(config)
        checksums = hashes(paths.values())
        for key, expected in (("head_checkpoint_path", HEAD_SHA256), ("head_config_path", CONFIG_SHA256),
                              ("backbone_checkpoint_path", BACKBONE_SHA256),
                              ("backbone_config_path", BACKBONE_CONFIG_SHA256)):
            if checksums[str(paths[key])] != expected:
                raise ValueError(f"Published asset checksum mismatch: {key}")
        args["model_args"].update(backbone_checkpoint_path=str(paths["backbone_checkpoint_path"]),
                                 backbone_cfg=str(paths["backbone_config_path"]))
        import os
        if any(k.startswith("LUNAR_EVAL_") for k in os.environ):
            raise ValueError("Remove LUNAR_EVAL_* overrides: the published config must remain fixed")
        with offline_imports(repo):
            runtime = import_local("terratorch_integration", repo)
            self.model = runtime.LunarObjectDetectionTask(**args)
            checkpoint = torch.load(paths["head_checkpoint_path"], map_location="cpu", weights_only=True, mmap=True)
            state = checkpoint.get("state_dict", checkpoint)
            if not isinstance(state, dict) or not any(k.startswith("model.") for k in state):
                raise ValueError("No published detector tensors in checkpoint")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            validate_model_keys(missing, unexpected)
            del state, checkpoint
            gc.collect()
            self.model.to(device="cuda:0", dtype=torch.float32).eval().requires_grad_(False)
        return {"code_revision": CODE_REVISION, "head_revision": HEAD_REVISION,
                "backbone_revision": BACKBONE_REVISION, "checkpoint_sha256": checksums,
                "missing_non_model_keys": missing, "unexpected_non_model_keys": unexpected,
                "precision": "float32", "quantization": None, "input_protocol": PROTOCOL,
                "scientific_result": False}

    def detect(self, image_path):
        if self.model is None:
            raise RuntimeError("Initialize detector first")
        pixels, preprocessing = prepare_png(image_path)
        tensor = self.torch.from_numpy(pixels).to(device="cuda:0", dtype=self.torch.float32)
        with self.torch.inference_mode():
            # Publisher validation/test applies task-level NMS after the model.
            output = self.model(tensor).output
            output = self.model.apply_nms_batch(output, 1)
        if not isinstance(output, (list, tuple)) or len(output) != 1:
            raise ValueError("Expected exactly one detector output")
        result = output[0]
        w, h = preprocessing["native_width"], preprocessing["native_height"]
        boxes = result["boxes"].float().cpu().tolist()
        boxes = [[x1*w/SIDE, y1*h/SIDE, x2*w/SIDE, y2*h/SIDE] for x1,y1,x2,y2 in boxes]
        parsed = readout(boxes, result["scores"].float().cpu().tolist(),
                         result["labels"].cpu().tolist(), w, h)
        return {**parsed, "preprocessing": preprocessing}

    def predict(self, request):
        raise NotImplementedError("Crater detector returns boxes; use detect + the declared fixed readout, not text QA")

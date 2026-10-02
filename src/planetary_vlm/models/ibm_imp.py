"""Published NASA–IBM frozen-backbone IMP segmentation system."""

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from time import perf_counter

from planetary_vlm.contracts import ModelResponse
from planetary_vlm.inference.runner import file_hash
from .local_runtime import checked_repo, hashes, import_local, local_path, offline_imports, release


def restore_imp_png(path, maximum=0.12):
    """Approximate reflectance from the documented 8-bit display transform.

    Clipping, quantization and lost nodata are irreversible. This is an
    exploratory PNG input track, not a reproduction using the original TIFFs.
    """
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.dtype != np.uint8 or pixels.shape != (256, 256):
        raise ValueError("IMP PNG track requires a grayscale uint8 256x256 image")
    return pixels.astype(np.float32) * (maximum / 255.0)


def validate_head_keys(missing, unexpected):
    # A frozen-head release may omit encoder tensors restored separately from
    # the backbone. Every neck/decoder/head tensor must come from the release.
    missing_head = [key for key in missing if key.startswith("model.") and not key.startswith("model.encoder.")]
    if missing_head or unexpected:
        raise ValueError(f"Incompatible published head: missing={missing_head}, unexpected={unexpected}")


def save_maps(directory, request_id, logits):
    """Persist raw two-class logits, probabilities and label IDs atomically."""
    import numpy as np
    from PIL import Image
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (2, 256, 256) or not np.isfinite(logits).all():
        raise ValueError("Expected finite IBM logits shaped (2,256,256)")
    scores = np.exp(logits - logits.max(axis=0, keepdims=True))
    probabilities = scores / scores.sum(axis=0, keepdims=True)
    labels = logits.argmax(axis=0).astype(np.uint8)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(request_id.encode()).hexdigest()
    arrays = BytesIO()
    np.savez_compressed(arrays, logits=logits, probabilities=probabilities)
    png = BytesIO()
    Image.fromarray(labels).save(png, format="PNG")
    files = {}
    for suffix, content in ((".npz", arrays.getvalue()), (".png", png.getvalue())):
        destination = directory / (name + suffix)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        files[destination.name] = hashlib.sha256(content).hexdigest()
    return {"output_type": "imp_segmentation", "classes": ["Background", "IMP"],
            "shape": [256, 256], "decision": "argmax_of_two_logits",
            "map_directory": str(directory.resolve()), "files_sha256": files,
            "label_png": name + ".png", "arrays_npz": name + ".npz"}


class IBMIMPAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = self.torch = None

    def preflight(self):
        if self.options.get("local_files_only", True) is not True:
            raise ValueError("Only offline inference is supported")
        repo = checked_repo(self.options, "terratorch_integration")
        paths = {key: local_path(self.options, key) for key in
                 ("head_config_path", "head_checkpoint_path", "backbone_config_path", "backbone_checkpoint_path")}
        if self.options.get("input_protocol") != "curated_imp_png_reflectance_0_0.12_v1":
            raise ValueError("Explicit exploratory IMP PNG input protocol is required")
        return repo, paths

    def initialize(self):
        repo, paths = self.preflight()
        import torch
        import yaml
        self.torch = torch
        device = self.options.get("device", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is requested but unavailable")
        torch.manual_seed(self.options.get("seed", 42))
        config = yaml.safe_load(paths["head_config_path"].read_text())
        if config["model"]["class_path"] != "terratorch_integration.LunarSegmentationTask":
            raise ValueError("Expected the published IMP LunarSegmentationTask configuration")
        normalization = config["data"]["init_args"]
        if normalization["means"] != [0.027375] or normalization["stds"] != [0.014783]:
            raise ValueError("IMP normalization differs from the audited release")
        args = deepcopy(config["model"]["init_args"])
        architecture = args["model_args"]
        if (architecture["backbone"] != "ni_lfm_v1_base" or architecture["backbone_modalities"] != ["nac"]
                or architecture["backbone_patch_size"] != 8 or architecture["num_classes"] != 2
                or architecture["decoder"] != "UNetDecoder" or args.get("freeze_backbone") is not True):
            raise ValueError("Expected the published frozen NAC/IMP head architecture")
        architecture["backbone_checkpoint_path"] = str(paths["backbone_checkpoint_path"])
        architecture["backbone_cfg"] = str(paths["backbone_config_path"])
        with offline_imports(repo):
            runtime = import_local("terratorch_integration", repo)
            self.model = runtime.LunarSegmentationTask(**args)
            checkpoint = torch.load(paths["head_checkpoint_path"], map_location="cpu", weights_only=True)
            state = checkpoint.get("state_dict", checkpoint)
            if not isinstance(state, dict) or not any(key.startswith("model.") for key in state):
                raise ValueError("Published checkpoint contains no segmentation model tensors")
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            validate_head_keys(missing, unexpected)
            self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        return {"checkpoint_sha256": hashes(paths.values()), "code_revision": self.options["code_revision"],
                "input_protocol": self.options["input_protocol"], "mean": 0.027375, "std": 0.014783,
                "precision": "float32", "device": device, "missing_encoder_keys": missing,
                "scientific_limitation": "PNG quantization/clipping and nodata loss; not native TIFF reproduction"}

    def predict(self, request):
        start = perf_counter()
        if len(request.image_paths) != 1 or request.modality_paths:
            return ModelResponse(request.request_id, "", "unsupported", 0, "Exactly one NAC intensity image is supported")
        if self.model is None:
            self.initialize()
        image = restore_imp_png(request.image_paths[0])
        tensor = self.torch.from_numpy(((image - 0.027375) / 0.014783)[None, None]).to(self.options.get("device", "cuda:0"))
        with self.torch.inference_mode():
            output = self.model(tensor).output
        if tuple(output.shape) != (1, 2, 256, 256):
            raise ValueError(f"Unexpected segmentation output: {tuple(output.shape)}")
        maps = save_maps(self.options["artifact_dir"], request.request_id, output[0].float().cpu().numpy())
        maps["input_protocol"] = self.options["input_protocol"]
        return ModelResponse(request.request_id, json.dumps(maps, ensure_ascii=False), "ok", perf_counter() - start)

    def close(self):
        release(self)


def verify_saved_maps(raw_response):
    payload = json.loads(raw_response)
    root = Path(payload["map_directory"]).resolve()
    for name, digest in payload["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Resume rejected: a saved IBM map is missing or changed")

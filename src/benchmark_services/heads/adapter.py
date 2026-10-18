"""Real frozen encoder + published local head, clean WAC input only, no targets."""
import json
import os
from pathlib import Path
import sys
import time

from planetary_vlm.contracts import ModelResponse
from planetary_vlm.heads_contract import read_deployment, pixel_hash
from ..common import settings as paths
from ..common.files import file_hash


class FrozenCraterAdapter:
    def __init__(self, options):
        self.options = dict(options)
        self.model = self.head = self.forward = None

    def initialize(self):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        root = paths.WORKSPACE_ROOT
        # Source checkout is required; no arbitrary runtime/repo supplied over HTTP.
        sys.path.insert(0, str(root))
        from linear_head_approach.encoders import load
        from linear_head_approach.heads import CraterHead
        from linear_head_approach.common import code_hash, rows, environment
        path, self.deployment = read_deployment(self.options["deployment_path"], self.options["deployment_sha256"], root)
        data = self.deployment
        if data["training_identity"]["code"] != code_hash():
            raise ValueError("Head training code differs from this checkout; restore its exact code")
        if data["training_identity"]["environment"] != environment():
            raise ValueError("Head service must use the same frozen environment as CLI training")
        saved = torch.load(path.parent / "head.pt", map_location="cpu", weights_only=True)
        if saved["identity"] != data["training_identity"] or saved["threshold"] != data["threshold"]:
            raise ValueError("Head checkpoint/deployment provenance mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for real encoder inference")
        self.model, self.forward, verified = load(Path(data["assets_dir"]), data["encoder"], root, "cuda")
        if verified != data["training_identity"]["feature_identity"]["assets"]:
            raise ValueError("Encoder asset identity differs from training")
        self.head = CraterHead(data["channels"], data["family"], data["hidden_channels"]).to("cuda").eval()
        self.head.load_state_dict(saved["head"], strict=True)
        self.head.requires_grad_(False)
        self.mean = saved["mean"].to("cuda")[None, :, None, None]
        self.std = saved["std"].to("cuda")[None, :, None, None]
        self.inputs = {row["rgb_pixels_sha256"]: row for row in rows(path.parent / "encoder_inputs.jsonl")}
        if any(r["split"] not in {"val", "test"} for r in self.inputs.values()):
            raise ValueError("Training samples cannot be served as benchmark scenes")
        return {"encoder": data["encoder"], "family": data["family"], "head_sha256": data["files_sha256"]["head.pt"],
                "deployment_sha256": self.options["deployment_sha256"], "threshold": data["threshold"],
                "clean_only": True, "timing": "live_encoder_plus_head_excludes_initial_loading", "input_warning": data["input_warning"]}

    def predict(self, request):
        import torch
        from torch.nn import functional as F
        from ..common.segmentation import save_crater_maps
        if self.model is None:
            raise RuntimeError("Initialize the head adapter first")
        if len(request.image_paths) != 1 or request.allowed_answers or request.do_sample:
            raise ValueError("This adapter produces crater masks, not question answers")
        image = paths.orbital_path(request.image_paths[0])
        row = self.inputs.get(pixel_hash(image))
        if row is None:
            raise ValueError("Only the admitted clean held-out WAC images are supported; no corruptions/train scenes")
        native = dict(request.modality_paths).get("vis")
        if not native or file_hash(paths.orbital_path(native)) != row["native_vis_sha256"]:
            raise ValueError("Paired native VIS input is missing or changed")
        input_row = row | {"rgb_image": image.relative_to(paths.WORKSPACE_ROOT).as_posix(),
                           "rgb_image_sha256": file_hash(image),
                           "native_vis": paths.orbital_path(native).relative_to(paths.WORKSPACE_ROOT).as_posix()}
        torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.inference_mode():
            feature = F.interpolate(self.forward(input_row), size=(24, 24), mode="bilinear", align_corners=False)
            logits = self.head((feature - self.mean) / self.std, 512)[0, 0].float().cpu().numpy()
        artifacts = save_crater_maps(self.options["artifact_dir"], request.request_id, logits, self.deployment["threshold"])
        torch.cuda.synchronize()
        return ModelResponse(request.request_id, json.dumps(artifacts), "ok", time.perf_counter() - begin)

    def close(self):
        self.model = self.head = self.forward = self.mean = self.std = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

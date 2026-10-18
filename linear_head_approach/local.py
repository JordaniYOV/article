"""Read-only local diagnostics and Windows/CLI stage coordination."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from .common import ROOT, config


def doctor():
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "workspace": str(ROOT),
              "workspace_free_gib": round(shutil.disk_usage(ROOT).free / 2**30, 2),
              "target": config()["hardware"], "gpu_model_smoke": "NOT_RUN"}
    try:
        import torch
        result.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result.update(gpu=torch.cuda.get_device_name(0), free_vram_gib=round(free / 2**30, 2), total_vram_gib=round(total / 2**30, 2))
    except (ImportError, OSError) as error:
        result["torch_error"] = f"{type(error).__name__}: {error}"
    result["note"] = "Environment/CUDA check only, NOT evidence that the encoders fit. Run smoke before full extraction."
    return result


def runtime(allow_download=False):
    from .common import git_revision
    from .encoders import RUNTIME_REV
    target = ROOT / "external/ibm-lfm"
    if not target.exists():
        if not allow_download:
            raise FileNotFoundError("IBM source missing. Run runtime --allow-download to fetch pinned CODE only")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model.git", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", RUNTIME_REV], check=True)
    if git_revision(target) != RUNTIME_REV:
        raise ValueError("Existing IBM source has another revision; it is not reset automatically")
    if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--", "ni_lfm", "terratorch_integration"], text=True).strip():
        raise ValueError("IBM source is dirty; preserve changes before using it")
    return {"runtime": str(target), "revision": RUNTIME_REV, "weights_downloaded": False}


def pipeline(encoder, experiment, assets, session, resume=False):
    """Sequential child processes release VRAM between extraction and each head."""
    cfg = config()
    experiment, assets = Path(experiment).resolve(), Path(assets).resolve()
    if any("data_rover" in {p.casefold() for p in path.parts} for path in (experiment, assets)):
        raise ValueError("Protected archive cannot contain outputs/assets")
    cache = experiment / "features" / encoder
    steps = [["audit"], ["assets", "--encoder", encoder, "--assets", str(assets)],
             ["extract", "--encoder", encoder, "--assets", str(assets), "--output", str(cache), "--session-id", session, "--smoke"],
             ["extract", "--encoder", encoder, "--assets", str(assets), "--output", str(cache), "--session-id", session],
             ["head-smoke", "--output", str(experiment / "software_checks")]]
    for family in cfg["heads"]:
        for seed in cfg["seeds"]:
            output = experiment / "runs" / encoder / family / f"seed_{seed}"
            steps.append(["fit", "--cache", str(cache), "--output", str(output), "--family", family, "--seed", str(seed)] + (["--resume"] if resume else []))
            steps.append(["evaluate", "--cache", str(cache), "--output", str(output), "--split", "val"])
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    logs = experiment / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(steps):
        command = [sys.executable, "-u", "-B", "-m", "linear_head_approach", *step]
        log_path = logs / f"{session}_{index:02d}_{step[0]}.log"
        with log_path.open("a", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    status = process.wait()
                    if status:
                        raise subprocess.CalledProcessError(status, command)
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                    raise
    return {"encoder": encoder, "experiment": str(experiment), "validation_completed": True,
            "test_evaluated": False, "test_target_usage": "integrity_audit_only",
            "next": "publish each frozen head, import deployment JSON in UI"}

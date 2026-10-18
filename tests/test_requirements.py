"""Offline checks for pip files, profile separation and CUDA wheel selection."""

from pathlib import Path
import re
import unittest

try:
    from packaging.requirements import Requirement
except ImportError:
    try:
        # Use pip's own parser when packaging is not installed separately.
        from pip._vendor.packaging.requirements import Requirement
    except ImportError:
        Requirement = None


ROOT = Path(__file__).resolve().parents[1]


def profile_lines(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Service files must be self-contained; no includes, indexes or options.
        if line.startswith("-"):
            raise ValueError("Requirements must contain standalone package entries")
        yield line


def colab_profile_lines(path, seen=None):
    """Colab profiles may reuse a pinned local profile; service ones stay standalone."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path.parent != (ROOT / "requirements").resolve() or path in seen:
        raise ValueError("Unsafe or cyclic requirement include")
    seen = seen | {path}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            yield from colab_profile_lines(path.parent / line[3:].strip(), seen)
        elif line.startswith("-"):
            raise ValueError("Only local -r includes are permitted in Colab profiles")
        else:
            yield line


@unittest.skipIf(Requirement is None, "Install requirements/api.txt for syntax checks")
class RequirementsTests(unittest.TestCase):
    def requirements(self, name):
        return [Requirement(line) for line in profile_lines(ROOT / name)]

    def test_every_requirement_file_has_valid_pip_entries(self):
        paths = list((ROOT / "requirements").iterdir())
        services = {"api.txt", "spacellava.txt", "ibm-imp.txt", "encoder-heads-windows.txt"}
        colab = {"ibm-crater-colab.txt", "ibm-val14-colab-lock.txt", "encoder-heads-colab.txt"}
        self.assertEqual({p.name for p in paths}, services | colab)
        self.assertFalse((ROOT / "requirements.txt").exists())
        for path in paths:
            with self.subTest(path=path.name):
                lines = profile_lines(path) if path.name in services else colab_profile_lines(path)
                self.assertTrue([Requirement(line) for line in lines])

    def test_encoder_head_profile_reuses_exact_frozen_colab_environment(self):
        direct = list(colab_profile_lines(ROOT / "requirements/ibm-val14-colab-lock.txt"))
        heads = list(colab_profile_lines(ROOT / "requirements/encoder-heads-colab.txt"))
        self.assertEqual(heads, direct)
        names = {Requirement(line).name.lower().replace("_", "-") for line in heads}
        self.assertTrue({"torch", "transformers", "safetensors", "rasterio", "numpy", "pillow"} <= names)

    def test_windows_heads_profile_omits_linux_cuda_and_llm_dependencies(self):
        requirements = self.requirements("requirements/encoder-heads-windows.txt")
        names = {r.name.lower() for r in requirements}
        self.assertFalse(names & {"triton", "bitsandbytes", "llava", "terratorch"})
        self.assertFalse(any(name.startswith("nvidia-") for name in names))
        self.assertTrue({"torch", "transformers", "rasterio", "hdf5plugin", "pandas", "sqlmodel", "fastapi"} <= names)
        torch = next(r for r in requirements if r.name == "torch")
        self.assertIn("2.12.0+cu130", torch.specifier)

    def test_spacellava_contains_full_runtime_not_ibm_or_restoration(self):
        names = {req.name.lower() for req in self.requirements("requirements/spacellava.txt")}
        self.assertTrue({"fastapi", "sqlmodel", "torch", "torchvision",
                         "transformers", "accelerate", "bitsandbytes", "sentencepiece",
                         "protobuf", "safetensors", "peft", "huggingface-hub"} <= names)
        self.assertNotIn("terratorch", names)
        self.assertNotIn("llava", names)  # local audited source, no obsolete pip pins
        self.assertFalse(names & {"pyarrow", "h5py", "hdf5plugin", "tifffile", "imagecodecs"})

    def test_ibm_contains_published_runtime_dependencies(self):
        names = {req.name.lower() for req in self.requirements("requirements/ibm-imp.txt")}
        self.assertTrue({"terratorch", "torchgeo", "timm", "lightning", "torchmetrics",
                         "omegaconf", "rasterio", "diffusers", "vector-quantize-pytorch",
                         "segmentation-models-pytorch", "pyyaml", "fastapi", "pyarrow"} <= names)
        self.assertNotIn("bitsandbytes", names)

    def test_each_windows_python_selects_one_cuda_wheel_per_package(self):
        for profile, cuda in (("spacellava", "cu128"), ("ibm-imp", "cu130")):
            requirements = self.requirements(f"requirements/{profile}.txt")
            for python in ("3.11", "3.12"):
                env = {"sys_platform": "win32", "python_version": python}
                urls = [r for r in requirements if r.url and r.marker.evaluate(env)]
                for package in ("torch", "torchvision"):
                    with self.subTest(profile=profile, python=python, package=package):
                        selected = [r for r in urls if r.name == package]
                        self.assertEqual(len(selected), 1)
                        url = selected[0].url
                        self.assertTrue(url.startswith("https://download.pytorch.org/whl/" + cuda + "/"))
                        self.assertIn("cp" + python.replace(".", ""), url)
                        self.assertTrue(url.endswith("win_amd64.whl"))

    def test_key_profile_constraints_have_a_shared_candidate(self):
        candidates = {
            "spacellava": {"numpy": "2.2.6",
                           "Pillow": "12.3.0", "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128"},
            "ibm-imp": {"numpy": "2.4.4", "pyarrow": "24.0.0", "hdf5plugin": "6.0.0",
                        "Pillow": "12.3.0", "torch": "2.12.0+cu130", "torchvision": "0.27.0+cu130"},
        }
        for profile, versions in candidates.items():
            for req in self.requirements(f"requirements/{profile}.txt"):
                if req.name in versions and not req.url:
                    with self.subTest(profile=profile, requirement=str(req)):
                        self.assertIn(versions[req.name], req.specifier)

    def test_api_contains_restoration_and_tests_not_model_runtime(self):
        names = {r.name.lower() for r in self.requirements("requirements/api.txt")}
        self.assertTrue({"fastapi", "sqlmodel", "python-multipart", "certifi", "numpy",
                         "pillow", "opencv-python-headless", "pyarrow", "h5py", "hdf5plugin",
                         "tifffile", "imagecodecs", "httpx", "packaging"} <= names)
        self.assertFalse(names & {"torch", "transformers", "terratorch", "bitsandbytes"})

    def test_each_service_contains_shared_pipeline_dependencies(self):
        expected = {"fastapi", "uvicorn", "sqlmodel", "sqlalchemy", "pydantic",
                    "pydantic-settings", "pillow", "numpy", "opencv-python-headless", "structlog"}
        for profile in ("api", "spacellava", "ibm-imp"):
            with self.subTest(profile=profile):
                names = {r.name.lower() for r in self.requirements(f"requirements/{profile}.txt")}
                self.assertTrue(expected <= names)
                # No duplicate version constraints except platform-specific CUDA URLs.
                packages = [r.name.lower() for r in self.requirements(f"requirements/{profile}.txt")
                            if not r.url]
                self.assertEqual(len(packages), len(set(packages)))

    def test_docker_install_commands_reference_existing_profiles(self):
        for filename in ("api.Dockerfile", "model.Dockerfile"):
            content = (ROOT / "deploy" / filename).read_text(encoding="utf-8")
            references = re.findall(r"-r (requirements/[\w-]+\.txt)", content)
            self.assertTrue(references)
            for path in references:
                with self.subTest(dockerfile=filename, profile=path):
                    self.assertTrue((ROOT / path).is_file())
        model = (ROOT / "deploy" / "model.Dockerfile").read_text(encoding="utf-8")
        mock_branch = model.split('if [ "$PROFILE" = "mock" ]; then', 1)[1].split("elif", 1)[0]
        self.assertEqual(re.findall(r"-r (requirements/[\w-]+\.txt)", mock_branch),
                         ["requirements/api.txt"])


if __name__ == "__main__":
    unittest.main()

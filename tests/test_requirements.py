"""Offline checks for pip files, profile separation and CUDA wheel selection."""

from pathlib import Path
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


def profile_lines(path, ancestors=()):
    path = path.resolve()
    if path in ancestors or not path.is_relative_to(ROOT):
        raise ValueError("Cyclic or out-of-workspace requirements include")
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            yield from profile_lines(path.parent / line[3:].strip(), (*ancestors, path))
        else:
            yield line


@unittest.skipIf(Requirement is None, "Install requirements/dev.txt for syntax checks")
class RequirementsTests(unittest.TestCase):
    def requirements(self, name):
        return [Requirement(line) for line in profile_lines(ROOT / name)]

    def test_every_requirement_file_has_valid_pip_entries(self):
        paths = [ROOT / "requirements.txt", *(ROOT / "requirements").glob("*.txt")]
        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(self.requirements(str(path.relative_to(ROOT))))

    def test_default_contains_full_spacellava_not_ibm(self):
        names = {req.name.lower() for req in self.requirements("requirements.txt")}
        self.assertTrue({"fastapi", "sqlmodel", "pyarrow", "h5py", "torch", "torchvision",
                         "transformers", "accelerate", "bitsandbytes", "sentencepiece",
                         "protobuf", "safetensors", "peft", "huggingface-hub"} <= names)
        self.assertNotIn("terratorch", names)
        self.assertNotIn("llava", names)  # local audited source, no obsolete pip pins

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
            "spacellava": {"numpy": "2.2.6", "pyarrow": "25.0.1", "hdf5plugin": "7.1.0",
                           "Pillow": "12.3.0", "torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128"},
            "ibm-imp": {"numpy": "2.4.4", "pyarrow": "24.0.0", "hdf5plugin": "6.0.0",
                        "Pillow": "12.3.0", "torch": "2.12.0+cu130", "torchvision": "0.27.0+cu130"},
        }
        for profile, versions in candidates.items():
            for req in self.requirements(f"requirements/{profile}.txt"):
                if req.name in versions and not req.url:
                    with self.subTest(profile=profile, requirement=str(req)):
                        self.assertIn(versions[req.name], req.specifier)

    def test_app_does_not_install_a_model_runtime(self):
        names = {r.name.lower() for r in self.requirements("requirements/app.txt")}
        self.assertFalse(names & {"torch", "transformers", "terratorch", "bitsandbytes"})


if __name__ == "__main__":
    unittest.main()

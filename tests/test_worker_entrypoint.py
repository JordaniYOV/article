"""Exercise real interpreter imports instead of masking src layout with sys.path."""

from importlib.util import find_spec
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "api" / "worker.py"


@unittest.skipIf(find_spec("sqlmodel") is None, "Install API dependencies")
class WorkerEntrypointTests(unittest.TestCase):
    def check_help(self, command):
        result = subprocess.run(command, cwd=ROOT / "tests", capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--once", result.stdout)
        self.assertIn("--help", result.stdout)

    def test_direct_file_launch_from_another_directory(self):
        self.check_help([sys.executable, "-I", str(WORKER), "--help"])

    def test_direct_file_finds_sibling_package_without_editable_install(self):
        # -S disables .pth/editable-install processing. Expose third-party
        # dependencies directly, but deliberately do not expose the src tree.
        dependency_paths = [p for p in sys.path if Path(p).name in {"site-packages", "dist-packages"}]
        bootstrap = (
            "import runpy,sys; "
            f"sys.path.extend({dependency_paths!r}); "
            "sys.argv=[sys.argv[1], '--help']; "
            "runpy.run_path(sys.argv[0], run_name='__main__')"
        )
        self.check_help([sys.executable, "-I", "-S", "-c", bootstrap, str(WORKER)])

    def test_module_launch_uses_normal_package_imports(self):
        # A normal module launch requires the project to be installed. The
        # direct-file test above covers checkouts with only dependencies present.
        try:
            distribution("planetary-vlm-benchmark")
        except PackageNotFoundError:
            self.skipTest("Install the project for python -m api.worker")
        self.check_help([sys.executable, "-I", "-m", "api.worker", "--help"])

    def test_src_module_launch_without_editable_install(self):
        dependency_paths = [p for p in sys.path if Path(p).name in {"site-packages", "dist-packages"}]
        bootstrap = (
            "import runpy,sys; "
            f"sys.path.extend({dependency_paths!r}); "
            f"sys.path.insert(0, {str(ROOT)!r}); "
            "sys.argv=['src.api.worker','--help']; "
            "runpy.run_module('src.api.worker',run_name='__main__')"
        )
        self.check_help([sys.executable, "-I", "-S", "-c", bootstrap])


if __name__ == "__main__":
    unittest.main()

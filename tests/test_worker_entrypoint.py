"""Selected-job worker help/import checks; never execute the real queue."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkerEntrypointTests(unittest.TestCase):
    def check_help(self, command):
        result = subprocess.run(command, cwd=ROOT / "tests", capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--run-id", result.stdout)
        self.assertIn("--mock", result.stdout)

    def test_both_direct_files_work_without_editable_install(self):
        for service in ("spacellava", "ibm"):
            with self.subTest(service=service):
                worker = ROOT / "src" / "benchmark_services" / service / "worker.py"
                self.check_help([sys.executable, "-I", "-S", str(worker), "--help"])

    def test_both_canonical_modules_have_selected_job_entrypoints(self):
        for service in ("spacellava", "ibm"):
            module = f"benchmark_services.{service}.worker"
            bootstrap = (f"import runpy,sys; sys.path.insert(0, {str(ROOT / 'src')!r}); "
                f"sys.argv=[{module!r}, '--help']; runpy.run_module({module!r}, run_name='__main__')")
            self.check_help([sys.executable, "-I", "-S", "-c", bootstrap])

    def test_old_unfiltered_worker_is_removed(self):
        self.assertFalse((ROOT / "src/api/worker.py").exists())


if __name__ == "__main__":
    unittest.main()

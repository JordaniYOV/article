"""Check the Windows benchmark instructions without executing their commands."""
import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


class WindowsRunbookTests(unittest.TestCase):
    def setUp(self):
        text = (ROOT / "linear_head_approach/gpu_runbook.md").read_text(encoding="utf-8")
        self.steps = "## 5." + text.split("## 5.", 1)[1]
        self.commands = re.findall(r"```powershell\n(.*?)```", self.steps, re.S)

    def test_benchmark_steps_need_no_bare_uv_or_system_python(self):
        for block in self.commands:
            self.assertNotRegex(block, r"(?m)^\s*(?:&\s+)?uv(?:\.exe)?\s")
            self.assertNotRegex(block, r"(?m)^\s*(?:python|py)(?:\.exe)?\s")
        self.assertIn("& $headPython -m venv .venv-api", self.steps)
        self.assertIn("& $apiPython -m pip install", self.steps)
        self.assertNotIn("& $headPython -m pip install", self.steps)

    def test_separate_environments_ports_and_clean_protocol_are_explicit(self):
        self.assertIn(r".\.venv-api\Scripts\python.exe -m uvicorn", self.steps)
        self.assertIn(r".\.venv-heads3060\Scripts\python.exe -m uvicorn", self.steps)
        self.assertIn("--port 8000", self.steps)
        self.assertIn("--port 8003", self.steps)
        self.assertIn("--port 5173 --strictPort", self.steps)
        self.assertIn("PLANETARY_API_SERVICE_MOCK=false", self.steps)
        self.assertIn("WAC_catalog_craters_heads", self.steps)
        self.assertIn("31 изображение", self.steps)
        for encoder in ("clip", "ibm"):
            for seed in (13, 37, 73):
                self.assertIn(f"{encoder}_spatial_{seed}", self.steps)

    @unittest.skipUnless(POWERSHELL, "PowerShell parser required")
    def test_all_powershell_blocks_parse_without_execution(self):
        encoded = base64.b64encode(json.dumps(self.commands).encode()).decode()
        harness = r"""
$ErrorActionPreference = 'Stop'
$snippets = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('__BLOCKS__')) | ConvertFrom-Json
foreach ($snippet in $snippets) {
    $tokens = $null
    $parseErrors = $null
    [System.Management.Automation.Language.Parser]::ParseInput($snippet, [ref]$tokens, [ref]$parseErrors) | Out-Null
    if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
}
""".replace("__BLOCKS__", encoded)
        result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", harness],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

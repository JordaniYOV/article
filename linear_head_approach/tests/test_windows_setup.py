"""Exercise the real PowerShell launcher functions with fake executables only."""
from pathlib import Path
import shutil
import subprocess
import unittest


POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
INSTALLER = Path(__file__).resolve().parents[1] / "setup_windows.ps1"


@unittest.skipUnless(POWERSHELL, "PowerShell required for the Windows installer checks")
class WindowsSetupTests(unittest.TestCase):
    def test_without_opt_in_makes_no_changes(self):
        completed = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                    "-File", str(INSTALLER)], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("No changes made", completed.stdout)
        self.assertNotIn("Using ", completed.stdout)

    def test_check_uv_only_probes_module_and_never_installs(self):
        script = str(INSTALLER).replace("'", "''")
        harness = r"""
$ErrorActionPreference = 'Stop'
function Get-Command {
    param($Name, $CommandType, $ErrorAction)
    if ($Name -eq 'py') { return [pscustomobject]@{Source = 'FakeUvPython'} }
}
function FakeUvPython {
    if (($args -join ',') -ne '-m,uv,--version') { throw 'Read-only check attempted installation' }
    $global:LASTEXITCODE = 0
}
& '__SCRIPT__' -CheckUv
""".replace("__SCRIPT__", script)
        completed = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                                    "-Command", harness], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Using py -m uv", completed.stdout)
        self.assertIn("No environment created", completed.stdout)

    def resolve(self, scenario):
        # Parse the production script and dot-source ONLY function definitions.
        # Never execute installer, uv, Python, pip, downloads, or GPU stages.
        script = str(INSTALLER).replace("'", "''")
        harness = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$tree = [System.Management.Automation.Language.Parser]::ParseFile('__SCRIPT__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($functionName in @('Test-UvLauncher', 'Resolve-UvLauncher')) {
    $definition = $tree.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $functionName}, $true)
    if (-not $definition) { throw "Missing function: $functionName" }
    . ([scriptblock]::Create($definition.Extent.Text))
}
$global:uvCalls = [System.Collections.Generic.List[string]]::new()
$global:uvAvailable = @{}
$global:uvCodes = @{}
function Get-Command {
    param($Name, $CommandType, $ErrorAction)
    if ($global:uvAvailable.ContainsKey($Name)) {
        return [pscustomobject]@{Source = $global:uvAvailable[$Name]}
    }
}
function FakeUv {
    $global:uvCalls.Add('uv:' + ($args -join ','))
    $global:LASTEXITCODE = $global:uvCodes['uv']
}
function FakePy {
    $global:uvCalls.Add('py:' + ($args -join ','))
    $global:LASTEXITCODE = $global:uvCodes['py']
}
function FakePython {
    $global:uvCalls.Add('python:' + ($args -join ','))
    $global:LASTEXITCODE = $global:uvCodes['python']
}
__SCENARIO__
""".replace("__SCRIPT__", script).replace("__SCENARIO__", scenario)
        completed = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", harness],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_native_uv_is_preferred(self):
        self.resolve(r"""
$global:uvAvailable = @{uv = 'FakeUv'; py = 'FakePy'}
$global:uvCodes = @{uv = 0; py = 0}
$launcher = Resolve-UvLauncher
if ($launcher.Executable -ne 'FakeUv' -or $launcher.Prefix.Count -ne 0) { throw 'Wrong native launcher' }
if (($global:uvCalls -join ';') -ne 'uv:--version') { throw 'Unexpected probes' }
""")

    def test_py_module_works_without_uv_in_path(self):
        self.resolve(r"""
$global:uvAvailable = @{py = 'FakePy'; python = 'FakePython'}
$global:uvCodes = @{py = 0; python = 0}
$launcher = Resolve-UvLauncher
if ($launcher.Executable -ne 'FakePy' -or ($launcher.Prefix -join ',') -ne '-m,uv') { throw 'Wrong py launcher' }
if (($global:uvCalls -join ';') -ne 'py:-m,uv,--version') { throw 'Unexpected probes' }
$uvArguments = @($launcher.Prefix) + @('venv', '--seed', '--python', '3.12.14', 'C:\path with spaces\.venv-heads3060')
& $launcher.Executable @uvArguments
if ($global:uvCalls[1] -ne 'py:-m,uv,venv,--seed,--python,3.12.14,C:\path with spaces\.venv-heads3060') { throw 'Arguments not preserved' }
""")

    def test_python_module_after_broken_uv_and_missing_py_module(self):
        self.resolve(r"""
$global:uvAvailable = @{uv = 'FakeUv'; py = 'FakePy'; python = 'FakePython'}
$global:uvCodes = @{uv = 1; py = 1; python = 0}
$launcher = Resolve-UvLauncher
if ($launcher.Executable -ne 'FakePython' -or ($launcher.Prefix -join ',') -ne '-m,uv') { throw 'Wrong python launcher' }
if (($global:uvCalls -join ';') -ne 'uv:--version;py:-m,uv,--version;python:-m,uv,--version') { throw 'Wrong fallback order' }
""")

    def test_missing_uv_fails_with_actionable_message(self):
        self.resolve(r"""
$global:uvAvailable = @{py = 'FakePy'}
$global:uvCodes = @{py = 1}
$failed = $false
try { $unused = Resolve-UvLauncher } catch {
    if ($_.Exception.Message -notlike '*py -m uv --version*') { throw 'Unhelpful missing-uv error' }
    $failed = $true
}
if (-not $failed) { throw 'Missing uv incorrectly admitted' }
""")

param([switch]$Install)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Install) {
    Write-Output 'No changes made. Use -Install to create .venv-heads3060 and install Windows dependencies (not model weights).'
    exit 0
}
if ($env:OS -ne 'Windows_NT') { throw 'This installer is for native Windows x64.' }
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: py -m pip install --user uv. Reopen the terminal, then rerun.'
}
$headEnvironment = Join-Path $projectRoot '.venv-heads3060'
$headPython = Join-Path $headEnvironment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $headPython)) {
    & uv venv --seed --python 3.12.14 $headEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12.14 environment.' }
}
& $headPython -c 'import sys; assert sys.version_info[:3] == (3,12,14), sys.version'
if ($LASTEXITCODE -ne 0) { throw 'Wrong Python version. Existing environment was not replaced.' }
$environmentRecords = Join-Path $projectRoot 'outputs/environments'
$frozenProfile = Join-Path $environmentRecords 'heads3060_frozen.txt'
$installProfile = Join-Path $projectRoot 'requirements/encoder-heads-windows.txt'
if (Test-Path -LiteralPath $frozenProfile) { $installProfile = $frozenProfile }
& $headPython -m pip install --no-cache-dir --only-binary=tokenizers --extra-index-url https://download.pytorch.org/whl/cu130 -r $installProfile
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Do not start training.' }
& $headPython -m pip install --no-cache-dir --no-deps -e $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed.' }
& $headPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed.' }
New-Item -ItemType Directory -Path $environmentRecords -Force | Out-Null
$frozenPackages = & $headPython -m pip freeze --all --exclude-editable
if ($LASTEXITCODE -ne 0) { throw 'Could not freeze installed packages.' }
if (Test-Path -LiteralPath $frozenProfile) {
    $previousPackages = Get-Content -LiteralPath $frozenProfile
    if (Compare-Object ($previousPackages | Sort-Object) ($frozenPackages | Sort-Object)) {
        throw 'Saved dependency lock differs from this environment; not overwriting it. Restore the same environment or use a new experiment.'
    }
} else { $frozenPackages | Set-Content -LiteralPath $frozenProfile -Encoding utf8 }
Push-Location -LiteralPath $projectRoot
try {
    & $headPython -B -m linear_head_approach doctor
    if ($LASTEXITCODE -ne 0) { throw 'Diagnostics failed.' }
} finally { Pop-Location }
Write-Output 'Installed environment only. No weights downloaded, no training started. Read linear_head_approach/gpu_runbook.md.'

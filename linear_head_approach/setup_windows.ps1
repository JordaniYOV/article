param([switch]$Install, [switch]$CheckUv)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot

function Test-UvLauncher {
    param([string]$Executable, [string[]]$Prefix)
    try {
        $probeArguments = @($Prefix) + @('--version')
        & $Executable @probeArguments *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Resolve-UvLauncher {
    # Use the Python module when pip installed uv outside PATH. Keep arguments
    # separate: executable paths may contain spaces; never use Invoke-Expression.
    $nativeUv = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($nativeUv -and (Test-UvLauncher -Executable $nativeUv.Source -Prefix @())) {
        return [pscustomobject]@{ Executable = $nativeUv.Source; Prefix = @(); Description = 'uv from PATH' }
    }
    foreach ($pythonName in @('py', 'python')) {
        $pythonLauncher = Get-Command $pythonName -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($pythonLauncher -and (Test-UvLauncher -Executable $pythonLauncher.Source -Prefix @('-m', 'uv'))) {
            return [pscustomobject]@{ Executable = $pythonLauncher.Source; Prefix = @('-m', 'uv'); Description = "$pythonName -m uv" }
        }
    }
    throw 'uv is unavailable via PATH, py -m uv, and python -m uv. Install it in your Python environment (py -m pip install --user uv), then verify py -m uv --version.'
}

if (-not $Install -and -not $CheckUv) {
    Write-Output 'No changes made. Use -CheckUv for a read-only uv check, or -Install to create .venv-heads3060 and install Windows dependencies (not model weights).'
    exit 0
}
if ($env:OS -ne 'Windows_NT') { throw 'This installer is for native Windows x64.' }
$uvLauncher = Resolve-UvLauncher
Write-Output "Using $($uvLauncher.Description): $($uvLauncher.Executable)"
if ($CheckUv -and -not $Install) {
    Write-Output 'uv check passed. No environment created, dependencies installed, or weights downloaded.'
    exit 0
}
$headEnvironment = Join-Path $projectRoot '.venv-heads3060'
$headPython = Join-Path $headEnvironment 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $headPython)) {
    $uvArguments = @($uvLauncher.Prefix) + @('venv', '--seed', '--python', '3.12.14', $headEnvironment)
    & $uvLauncher.Executable @uvArguments
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

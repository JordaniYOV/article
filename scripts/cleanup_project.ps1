param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$taskReport = Join-Path $taskRoot 'outputs\cleanup_20261019'
$taskSkip = @('data_rover','.git','.codex','.state','node_modules','__pycache__')
if ($Apply -and (Test-Path -LiteralPath (Join-Path $taskReport 'summary.json'))) {
    throw 'This cleanup already completed. Preserve its journal; do not run it again.'
}

function Assert-TaskPath([string]$taskPath) {
    $taskFull = [IO.Path]::GetFullPath($taskPath)
    if (-not $taskFull.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw "Outside workspace: $taskFull" }
    $taskRelative = $taskFull.Substring($taskRoot.Length + 1)
    if (($taskRelative.Split('\') | Where-Object { $_ -in @('data_rover','.git','.codex','.state') }).Count) { throw "Protected path: $taskFull" }
    $taskCursor = $taskFull
    while ($taskCursor -ne $taskRoot) {
        if (Test-Path -LiteralPath $taskCursor) {
            if (((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point: $taskCursor" }
        }
        $taskCursor = [IO.Path]::GetDirectoryName($taskCursor)
    }
    return $taskFull
}

function Get-TaskFiles([string]$taskPath) {
    foreach ($taskItem in Get-ChildItem -LiteralPath $taskPath -Force) {
        if (($taskItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
        if ($taskItem.PSIsContainer) {
            if ($taskItem.Name -in $taskSkip -or $taskItem.Name -like '.venv*') { continue }
            Get-TaskFiles $taskItem.FullName
        } else { $taskItem }
    }
}

function Save-TaskJson([string]$taskPath, $taskValue) {
    $taskPath = Assert-TaskPath $taskPath
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskPath)) | Out-Null
    $taskText = ConvertTo-Json -InputObject $taskValue -Depth 60
    [IO.File]::WriteAllText($taskPath, $taskText + "`n", [Text.UTF8Encoding]::new($false))
}

function Save-TaskBytes([string]$taskPath, [byte[]]$taskBytes, [string]$taskHash) {
    $taskPath = Assert-TaskPath $taskPath
    if (Test-Path -LiteralPath $taskPath) {
        if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLower() -ne $taskHash) { throw "Preserved file differs: $taskPath" }
    } else {
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($taskPath)) | Out-Null
        [IO.File]::WriteAllBytes($taskPath, $taskBytes)
    }
    if ((Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLower() -ne $taskHash) { throw "Preservation hash mismatch: $taskPath" }
}

function Read-TaskZipMember($taskZip, [string]$taskMember) {
    if ($taskMember.Contains('..') -or $taskMember.Contains('\') -or $taskMember.StartsWith('/')) { throw 'Unsafe member' }
    $taskEntry = $taskZip.GetEntry($taskMember)
    if ($null -eq $taskEntry -or $taskEntry.Length -gt 2MB) { throw "Missing or excessive member: $taskMember" }
    $taskStream = $taskEntry.Open()
    $taskMemory = [IO.MemoryStream]::new()
    try { $taskStream.CopyTo($taskMemory); return ,$taskMemory.ToArray() }
    finally { $taskStream.Dispose(); $taskMemory.Dispose() }
}

$taskDirectories = @(
    'data_orbital\.crater_seg_initial_attempt_20261018',
    'data_orbital\orbital_300_v3',
    'data_orbital\crater_annotations_v1',
    'data_orbital\crater_annotations_v2',
    'data_orbital\crater_annotations_v3',
    'data_orbital\crater_annotations_v4',
    'data_orbital\cvat_images',
    'data_orbital\orbital_questions_v1',
    'outputs\archives\linear_head_approach_before_neural_v2_20261018\encoder_probe_wac_v1'
) | ForEach-Object { Assert-TaskPath (Join-Path $taskRoot $_) } | Where-Object { Test-Path -LiteralPath $_ }
$taskInventory = @(Get-TaskFiles $taskRoot)
$taskArtifacts = @($taskInventory | Where-Object { $_.Extension -in @('.ipynb','.zip') })
$taskCovered = @{}
foreach ($taskDirectory in $taskDirectories) {
    foreach ($taskFile in Get-TaskFiles $taskDirectory) { $taskCovered[$taskFile.FullName] = $true }
}
foreach ($taskFile in $taskArtifacts) { $taskCovered[$taskFile.FullName] = $true }
$taskPlan = [ordered]@{
    workspace=$taskRoot; protected_archive_accessed=$false; recycle_bin=$true
    notebooks=@($taskArtifacts | Where-Object Extension -eq '.ipynb').Count
    zip_files=@($taskArtifacts | Where-Object Extension -eq '.zip').Count
    dataset_directories=@($taskDirectories | ForEach-Object { $_.Substring($taskRoot.Length+1) })
    removed_files=$taskCovered.Count
    removed_bytes=[long](($taskInventory | Where-Object { $taskCovered.ContainsKey($_.FullName) } | Measure-Object Length -Sum).Sum)
    retained=@('head_training_v2','crater_seg_100_v1','orbital_300_v4','crater_annotations_v5','crater_source_annotations_v1','source_comparison_v1','hash-bound provenance','model runs and weights')
}
if (-not $Apply) { $taskPlan | ConvertTo-Json -Depth 6; return }

# Preserve the old manifest bytes and any asset which is not identical in v4.
$taskV3 = Join-Path $taskRoot 'data_orbital\orbital_300_v3'
if (Test-Path -LiteralPath $taskV3) {
    foreach ($taskFile in Get-TaskFiles $taskV3) {
        $taskSuffix = $taskFile.FullName.Substring($taskV3.Length+1)
        $taskCurrent = Join-Path $taskRoot ('data_orbital\orbital_300_v4\' + $taskSuffix)
        $taskHash = (Get-FileHash -LiteralPath $taskFile.FullName -Algorithm SHA256).Hash.ToLower()
        if ($taskFile.Extension -notin @('.json','.jsonl') -and (Test-Path -LiteralPath $taskCurrent) -and
            (Get-FileHash -LiteralPath $taskCurrent -Algorithm SHA256).Hash.ToLower() -eq $taskHash) { continue }
        Save-TaskBytes (Join-Path $taskRoot ('data_orbital\provenance\orbital_300_v3\' + $taskSuffix)) ([IO.File]::ReadAllBytes($taskFile.FullName)) $taskHash
    }
}

# Preserve original CVAT XML; sealed archive and XML hashes are not rewritten.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskAnnotation = Join-Path $taskRoot 'data_orbital\crater_annotations_v5'
$taskMetadata = Get-Content -LiteralPath (Join-Path $taskAnnotation 'metadata.json') -Raw | ConvertFrom-Json
$taskPreservedArchives = @()
foreach ($taskExport in $taskMetadata.exports.PSObject.Properties) {
    $taskArchivePath = Join-Path $taskAnnotation ('exports\' + $taskExport.Name)
    if (-not (Test-Path -LiteralPath $taskArchivePath)) { continue }
    if ((Get-FileHash -LiteralPath $taskArchivePath -Algorithm SHA256).Hash.ToLower() -ne $taskExport.Value.archive_sha256) { throw 'CVAT archive changed' }
    $taskZip = [IO.Compression.ZipFile]::OpenRead($taskArchivePath)
    try {
        if ($taskZip.Entries.Count -ne 1) { throw 'Unexpected additional CVAT payload' }
        $taskBytes = Read-TaskZipMember $taskZip 'annotations.xml'
        $taskXml = Join-Path $taskAnnotation ('exports\' + [IO.Path]::GetFileNameWithoutExtension($taskExport.Name) + '\annotations.xml')
        Save-TaskBytes $taskXml $taskBytes $taskExport.Value.xml_sha256
        $taskPreservedArchives += @{archive=$taskExport.Name;original_archive_sha256=$taskExport.Value.archive_sha256;xml_sha256=$taskExport.Value.xml_sha256;xml=$taskXml.Substring($taskRoot.Length+1);original_archive_verified_before_cleanup=$true}
    } finally { $taskZip.Dispose() }
}
if ($taskPreservedArchives.Count) { Save-TaskJson (Join-Path $taskAnnotation 'exports\unpacked_provenance.json') @{archives=$taskPreservedArchives;original_zip_bytes_retained=$false} }

# Retain only the 26 publisher CSVs actually used by the preserved annotations.
$taskSourceRoot = Join-Path $taskRoot 'data_orbital\crater_source_annotations_v1'
$taskMarsZip = Join-Path $taskSourceRoot 'mars_crater_metadata.zip'
if (Test-Path -LiteralPath $taskMarsZip) {
    $taskArchiveHash = (Get-FileHash -LiteralPath $taskMarsZip -Algorithm SHA256).Hash.ToLower()
    if ($taskArchiveHash -ne 'fe92462495dd59fde79ca773031abc78b35640ab9ac9242ac65d9a959e2306e4') { throw 'Publisher metadata ZIP changed' }
    $taskZip = [IO.Compression.ZipFile]::OpenRead($taskMarsZip)
    try {
        $taskMembers = @()
        foreach ($taskLine in Get-Content -LiteralPath (Join-Path $taskAnnotation 'samples.jsonl')) {
            $taskSample = $taskLine | ConvertFrom-Json
            $taskAudit = $taskSample.source_annotation_audit
            if (-not $taskAudit.source_zip_member) { continue }
            $taskBytes = Read-TaskZipMember $taskZip $taskAudit.source_zip_member
            $taskCsv = Join-Path $taskSourceRoot ('mars_crater_metadata\' + $taskAudit.source_zip_member.Replace('/','\'))
            Save-TaskBytes $taskCsv $taskBytes $taskAudit.source_zip_member_sha256
            $taskMembers += @{sample_id=$taskSample.sample_id;member=$taskAudit.source_zip_member;sha256=$taskAudit.source_zip_member_sha256;path=$taskCsv.Substring($taskRoot.Length+1)}
        }
        Save-TaskJson (Join-Path $taskSourceRoot 'unpacked_mars_metadata.json') @{original_archive_sha256=$taskArchiveHash;selected_members=$taskMembers;original_archive_verified_before_cleanup=$true;original_zip_bytes_retained=$false;full_archive_available=$false}
    } finally { $taskZip.Dispose() }
}

# Produce recovery metadata from the retained, unmodified six v4 manifests.
$taskTracks = @{}
foreach ($taskPlanet in @('mars','moon')) {
    $taskTracks[$taskPlanet] = @{}
    foreach ($taskTrack in @('segmentation','height','train')) {
        $taskManifest = Join-Path $taskRoot "data_orbital\orbital_300_v4\$taskPlanet\${taskTrack}_manifest.jsonl"
        $taskTracks[$taskPlanet][$taskTrack] = @(Get-Content -LiteralPath $taskManifest | ForEach-Object { $_ | ConvertFrom-Json })
    }
}
Save-TaskJson (Join-Path $taskRoot 'configs\datasets\orbital_builtin_recovery_v4.json') @{version='orbital-300-v4';tracks=$taskTracks}

# Record the exact plan before deletion; never erase source identities or runs.
Save-TaskJson (Join-Path $taskReport 'plan.json') $taskPlan
$taskRemovalRecords = @($taskInventory | Where-Object { $taskCovered.ContainsKey($_.FullName) } | ForEach-Object {
    @{path=$_.FullName.Substring($taskRoot.Length+1);bytes=$_.Length;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()}
})
Save-TaskJson (Join-Path $taskReport 'removed_files.json') $taskRemovalRecords
Add-Type -AssemblyName Microsoft.VisualBasic
$taskRemoved = @()
foreach ($taskDirectory in $taskDirectories) {
    $taskValidated = Assert-TaskPath $taskDirectory
    # Verify descendants before invoking a recursive filesystem operation.
    foreach ($taskDescendant in Get-TaskFiles $taskValidated) { Assert-TaskPath $taskDescendant.FullName | Out-Null }
    $taskReparse = @(Get-ChildItem -LiteralPath $taskValidated -Recurse -Force | Where-Object { ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 })
    if ($taskReparse.Count) { throw 'Recursive target contains a reparse point' }
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($taskValidated,[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,[Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
    $taskRemoved += $taskValidated.Substring($taskRoot.Length+1)
    Save-TaskJson (Join-Path $taskReport 'completed_targets.json') $taskRemoved
}
foreach ($taskArtifact in $taskArtifacts) {
    if (-not (Test-Path -LiteralPath $taskArtifact.FullName)) { continue }
    $taskValidated = Assert-TaskPath $taskArtifact.FullName
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($taskValidated,[Microsoft.VisualBasic.FileIO.UIOption]::OnlyErrorDialogs,[Microsoft.VisualBasic.FileIO.RecycleOption]::SendToRecycleBin,[Microsoft.VisualBasic.FileIO.UICancelOption]::ThrowException)
    $taskRemoved += $taskValidated.Substring($taskRoot.Length+1)
    Save-TaskJson (Join-Path $taskReport 'completed_targets.json') $taskRemoved
}
$taskRemaining = @(Get-TaskFiles $taskRoot | Where-Object { $_.Extension -in @('.ipynb','.zip') })
if ($taskRemaining.Count) { throw 'Some notebook/ZIP artifacts remain' }
Save-TaskJson (Join-Path $taskReport 'summary.json') ($taskPlan + @{complete=$true;remaining_project_notebooks_and_zip=0})
$taskPlan | ConvertTo-Json -Depth 6

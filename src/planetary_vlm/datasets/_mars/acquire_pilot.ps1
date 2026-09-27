# Bounded NASA PDS acquisition using Windows system certificate validation.
# Six metadata labels; optionally the six corresponding source images (< 60 MB).
param([switch]$Images, [Parameter(Mandatory)][string]$DataRoot)
$ErrorActionPreference = 'Stop'
$stereoOutput = Join-Path ([IO.Path]::GetFullPath($DataRoot)) 'raw/mastcam_stereo_pilot_v1'
New-Item -ItemType Directory -Path $stereoOutput -Force | Out-Null
$stereoCandidates = @(
    @{volume='0007';sol='0703';id='0703ML0029750010304398E01_DXXX'},
    @{volume='0007';sol='0703';id='0703MR0029750000402286E01_DXXX'},
    @{volume='0008';sol='0753';id='0753ML0032370030400004E01_DXXX'},
    @{volume='0008';sol='0753';id='0753MR0032370090403732E01_DXXX'},
    @{volume='0008';sol='0788';id='0788ML0034350010400503E01_DXXX'},
    @{volume='0008';sol='0788';id='0788MR0034350000500041E01_DXXX'}
)
$stereoRecords = @()
foreach ($stereoCandidate in $stereoCandidates) {
    $stereoBase = 'https://planetarydata.jpl.nasa.gov/img/data/msl/msl_mmm/data_MSLMST/volume_' + $stereoCandidate.volume + '_raw/SURFACE/' + $stereoCandidate.sol + '/'
    $stereoName = $stereoCandidate.id.ToLowerInvariant() + '.xml'
    $stereoPath = Join-Path $stereoOutput $stereoName
    $stereoUrl = $stereoBase + $stereoName
    if (-not (Test-Path -LiteralPath $stereoPath)) {
        $stereoResponse = Invoke-WebRequest -Uri $stereoUrl -TimeoutSec 30
        $stereoText = [string]$stereoResponse.Content
        if ([Text.Encoding]::UTF8.GetByteCount($stereoText) -gt 1000000) { throw 'Label exceeded acquisition limit' }
        [xml]$stereoXml = $stereoText
        if (-not $stereoXml.SelectSingleNode("//*[local-name()='alternate_id' and text()='$($stereoCandidate.id)']")) { throw 'Source ID mismatch' }
        [IO.File]::WriteAllText($stereoPath, $stereoText, [Text.UTF8Encoding]::new($false))
    }
    [xml]$stereoXml = Get-Content -LiteralPath $stereoPath -Raw
    $stereoFile = $stereoXml.SelectSingleNode("//*[local-name()='File_Area_Observational']/*[local-name()='File']")
    $stereoImageName = $stereoFile.SelectSingleNode("*[local-name()='file_name']").InnerText
    $stereoImageSize = [long]$stereoFile.SelectSingleNode("*[local-name()='file_size']").InnerText
    if ($stereoImageName -ne ($stereoCandidate.id + '.IMG') -or $stereoImageSize -gt 10000000) { throw 'Image identity/size limit failed' }
    $stereoImagePath = Join-Path $stereoOutput $stereoImageName
    if ($Images -and -not (Test-Path -LiteralPath $stereoImagePath)) {
        Invoke-WebRequest -Uri ($stereoBase + $stereoImageName) -TimeoutSec 30 -OutFile $stereoImagePath
    }
    if (Test-Path -LiteralPath $stereoImagePath) {
        if ((Get-Item -LiteralPath $stereoImagePath).Length -ne $stereoImageSize) { throw 'Image size differs from PDS label' }
    }
    $stereoRecords += [ordered]@{
        product_id=$stereoCandidate.id; label_url=$stereoUrl
        label_sha256=(Get-FileHash -LiteralPath $stereoPath -Algorithm SHA256).Hash.ToLowerInvariant()
        image_url=($stereoBase + $stereoImageName); image_bytes=$stereoImageSize
        image_downloaded=(Test-Path -LiteralPath $stereoImagePath)
        status='source_metadata_only_not_a_confirmed_stereo_pair'
    }
    Write-Output ('Checked ' + $stereoCandidate.id)
}
$stereoManifestPath = Join-Path $stereoOutput 'acquisition.json'
[IO.File]::WriteAllText($stereoManifestPath, ($stereoRecords | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))

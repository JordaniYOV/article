# Metadata-only discovery; no IMG/JPG or full dataset downloads.
param([ValidateSet('Catalog','Labels','Aliases','Images')][string]$Stage='Catalog', [Parameter(Mandatory)][string]$DataRoot)
$ErrorActionPreference = 'Stop'
$scanData = [IO.Path]::GetFullPath($DataRoot)
$scanOut = Join-Path $scanData 'raw/mastcam_partner_search_v1'
New-Item -ItemType Directory -Path $scanOut -Force | Out-Null
$scanBase = 'https://planetarydata.jpl.nasa.gov/img/data/msl/msl_mmm/data_MSLMST/'
if ($Stage -eq 'Catalog') {
    $scanRows = Get-Content (Join-Path $scanData 'interim/mars_bench_msl_v1/test/provenance.jsonl') | ForEach-Object { $_ | ConvertFrom-Json }
    $scanSols = @($scanRows | ForEach-Object { $_.mission_image_id.Substring(0,4) } | Sort-Object -Unique)
    $scanRootPath = Join-Path $scanOut 'root.html'
    if (-not (Test-Path $scanRootPath)) { Invoke-WebRequest $scanBase -OutFile $scanRootPath -TimeoutSec 30 }
    $scanVolumes = @([regex]::Matches((Get-Content $scanRootPath -Raw), 'href="(volume_\d{4}_raw/)"') | ForEach-Object {$_.Groups[1].Value} | Sort-Object -Unique)
    $scanSurfaces = $scanVolumes | ForEach-Object -Parallel {
        $p = Join-Path $using:scanOut ($_.TrimEnd('/') + '_surface.html')
        $url = $using:scanBase + $_ + 'SURFACE/'
        try {
            if (-not (Test-Path $p)) { Invoke-WebRequest $url -OutFile $p -TimeoutSec 30 }
            $html = Get-Content $p -Raw
            foreach ($m in [regex]::Matches($html, 'href="(\d{4})/"')) {
                if ($m.Groups[1].Value -in $using:scanSols) {
                    [pscustomobject]@{sol=$m.Groups[1].Value;url=($url+$m.Groups[1].Value+'/');volume=$_}
                }
            }
        } catch { [pscustomobject]@{error=$_.Exception.Message;url=$url} }
    } -ThrottleLimit 4
    $scanSurfaces = @($scanSurfaces | Sort-Object url -Unique)
    $scanSurfaces | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $scanOut 'surfaces.json') -Encoding utf8
    Write-Output ('Located sol directories: ' + @($scanSurfaces | Where-Object sol).Count)
    $scanCatalog = $scanSurfaces | Where-Object sol | ForEach-Object -Parallel {
        $p = Join-Path $using:scanOut ($_.volume.TrimEnd('/') + '_' + $_.sol + '.html')
        try {
            if (-not (Test-Path $p)) { Invoke-WebRequest $_.url -OutFile $p -TimeoutSec 30 }
            if ((Get-Item $p).Length -gt 20000000) { throw 'Directory page exceeded 20 MB limit' }
            $ids = @([regex]::Matches((Get-Content $p -Raw), 'href="(\d{4}M[LR]\d+E\d+_DXXX\.xml)"', 'IgnoreCase') | ForEach-Object {$_.Groups[1].Value} | Sort-Object -Unique)
            [pscustomobject]@{sol=$_.sol;url=$_.url;volume=$_.volume;label_names=$ids;directory_sha256=(Get-FileHash $p -Algorithm SHA256).Hash.ToLowerInvariant()}
        } catch { [pscustomobject]@{sol=$_.sol;url=$_.url;error=$_.Exception.Message} }
    } -ThrottleLimit 4
    $scanCatalog | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $scanOut 'catalog.json') -Encoding utf8
    Write-Output ('Catalogued directories: ' + @($scanCatalog | Where-Object label_names).Count)
    Write-Output ('Errors: ' + @($scanCatalog | Where-Object error).Count)
} elseif ($Stage -in @('Labels','Aliases')) {
    $scanRequests = Get-Content (Join-Path $scanOut 'label_requests.json') -Raw | ConvertFrom-Json
    if ($scanRequests.Count -gt 12000) { throw 'Refusing more than 12000 metadata labels in this batch' }
    $scanLabelDir = Join-Path $scanOut 'labels'
    New-Item -ItemType Directory -Path $scanLabelDir -Force | Out-Null
    if ($Stage -eq 'Aliases') {
        $scanRequests = @($scanRequests | Where-Object { $_.priority -eq 0 -and -not (Test-Path (Join-Path $scanLabelDir $_.name)) })
        if ($scanRequests.Count -eq 0) { Write-Output 'No additional identity labels required'; exit 0 }
    }
    $scanClient = [System.Net.Http.HttpClient]::new()
    $scanClient.Timeout = [TimeSpan]::FromSeconds(30)
    $scanResults = $scanRequests | ForEach-Object -Parallel {
        $p = Join-Path $using:scanLabelDir $_.name
        try {
            if (-not (Test-Path $p)) {
                $client = $using:scanClient
                $bytes = $client.GetByteArrayAsync([string]$_.url).GetAwaiter().GetResult()
                if ($bytes.Length -gt 1000000) { throw 'Label exceeded 1 MB limit' }
                [IO.File]::WriteAllBytes($p, $bytes)
            }
            if ((Get-Item $p).Length -gt 1000000) { throw 'Label exceeded 1 MB limit' }
            [xml]$xml = Get-Content $p -Raw
            if (-not $xml.SelectSingleNode("//*[local-name()='alternate_id']")) { throw 'Missing source identity' }
            [pscustomobject]@{name=$_.name;url=$_.url;status='downloaded';sha256=(Get-FileHash $p -Algorithm SHA256).Hash.ToLowerInvariant()}
        } catch { [pscustomobject]@{name=$_.name;url=$_.url;status='error';error=$_.Exception.Message} }
    } -ThrottleLimit 4
    $scanClient.Dispose()
    $manifestName = if ($Stage -eq 'Aliases') { 'label_acquisition_aliases.json' } else { 'label_acquisition.json' }
    $scanResults | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $scanOut $manifestName) -Encoding utf8
    Write-Output ('Labels checked: ' + $scanResults.Count + '; errors: ' + @($scanResults | Where-Object status -eq 'error').Count)
} else {
    $scanImages = Get-Content (Join-Path $scanOut 'verification_requests.json') -Raw | ConvertFrom-Json
    if ($scanImages.Count -gt 12) { throw 'Pilot limited to twelve images' }
    $scanImageDir = Join-Path $scanData 'raw/mastcam_pair_verification_v1'
    New-Item -ItemType Directory -Path $scanImageDir -Force | Out-Null
    $scanBytes = 0
    foreach ($r in $scanImages) {
        $labelPath = Join-Path $scanOut ('labels/' + $r.label_name)
        [xml]$xml = Get-Content $labelPath -Raw
        $file = $xml.SelectSingleNode("//*[local-name()='File_Area_Observational']/*[local-name()='File']")
        $name = $file.SelectSingleNode("*[local-name()='file_name']").InnerText
        $size = [long]$file.SelectSingleNode("*[local-name()='file_size']").InnerText
        if ($name -ne $r.image_name -or $name -ne ($r.product_id + '.IMG') -or $size -gt 10000000) { throw 'Image identity/size limit failed' }
        $scanBytes += $size
        if ($scanBytes -gt 100000000) { throw 'Pilot exceeded 100 MB limit' }
        Copy-Item -LiteralPath $labelPath -Destination (Join-Path $scanImageDir $r.label_name)
        $path = Join-Path $scanImageDir $name
        if (-not (Test-Path $path)) { Invoke-WebRequest $r.image_url -OutFile $path -TimeoutSec 30 }
        if ((Get-Item $path).Length -ne $size) { throw 'Downloaded image size mismatch' }
        Write-Output ('Pilot image checked: ' + $r.product_id)
    }
    Write-Output ('Pilot bytes: ' + $scanBytes)
}

param([Parameter(Mandatory)][string]$Plan, [Parameter(Mandatory)][string]$Cache)
$ErrorActionPreference = 'Stop'
$batchPlan = Get-Content -LiteralPath $Plan -Raw | ConvertFrom-Json
$batchCache = [IO.Path]::GetFullPath($Cache)
New-Item -ItemType Directory -Path $batchCache -Force | Out-Null
if ($batchPlan.images.Count -gt 3000 -or $batchPlan.planned_image_bytes -gt 8000000000) { throw 'Batch exceeds 3000 images / 8 GB; review plan' }
$batchClient = [System.Net.Http.HttpClient]::new()
$batchClient.Timeout = [TimeSpan]::FromSeconds(45)
$batchResults = $batchPlan.images | ForEach-Object -Parallel {
    $r = $_
    try {
        if ($r.image_name -ne ($r.product_id + '.IMG') -or $r.image_bytes -gt 10000000) { throw 'Image identity / 10 MB bound failed' }
        if (-not ([string]$r.image_url).StartsWith('https://planetarydata.jpl.nasa.gov/img/data/msl/')) { throw 'Unexpected source host' }
        $path = Join-Path $using:batchCache $r.image_name
        if (Test-Path -LiteralPath $path) {
            if ((Get-Item -LiteralPath $path).Length -ne $r.image_bytes) { throw 'Cached image size mismatch; retained for inspection' }
        } else {
            $client = $using:batchClient
            $bytes = $client.GetByteArrayAsync([string]$r.image_url).GetAwaiter().GetResult()
            if ($bytes.Length -ne $r.image_bytes) { throw 'Downloaded image size mismatch' }
            [IO.File]::WriteAllBytes($path, $bytes)
        }
        [pscustomobject]@{product_id=$r.product_id;status='downloaded';url=$r.image_url;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$r.image_bytes}
    } catch { [pscustomobject]@{product_id=$r.product_id;status='error';url=$r.image_url;error=$_.Exception.Message} }
} -ThrottleLimit 4
$batchClient.Dispose()
$batchResults | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $batchCache 'acquisition.json') -Encoding utf8
Write-Output ('Images checked: ' + $batchResults.Count + '; errors: ' + @($batchResults | Where-Object status -eq 'error').Count)

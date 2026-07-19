param(
    [Parameter(Mandatory = $true)][string]$SourceFile,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [Parameter(Mandatory = $true)][hashtable]$RegionFileMap,
    [string]$FileHeader = "",
    [string]$FileFooter = "}"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$lines = Get-Content -LiteralPath $SourceFile
$regions = @{}
$currentRegion = $null
$regionStart = -1

for ($i = 0; $i -lt $lines.Count; $i++) {
    $line = $lines[$i]
    if ($line -match '^\s*#region\s+(.+)$') {
        if ($null -ne $currentRegion) {
            throw "Nested region at line $($i + 1): $currentRegion"
        }
        $currentRegion = $Matches[1].Trim()
        $regionStart = $i + 1
        continue
    }

    if ($line -match '^\s*#endregion') {
        if ($null -eq $currentRegion) {
            throw "Unmatched #endregion at line $($i + 1)"
        }
        $regions[$currentRegion] = $lines[$regionStart..($i - 1)]
        $currentRegion = $null
        $regionStart = -1
    }
}

foreach ($regionName in $RegionFileMap.Keys) {
    if (-not $regions.ContainsKey($regionName)) {
        throw "Region '$regionName' not found in $SourceFile. Available: $($regions.Keys -join ', ')"
    }
}

foreach ($entry in $RegionFileMap.GetEnumerator()) {
    $regionName = $entry.Key
    $fileName = $entry.Value
    $body = ($regions[$regionName] -join [Environment]::NewLine).TrimEnd()
    $content = @(
        $FileHeader
        $body
        $FileFooter
    ) -join [Environment]::NewLine
    $target = Join-Path $OutputDirectory $fileName
    Set-Content -LiteralPath $target -Value $content -Encoding UTF8
    Write-Host "Wrote $target ($($regions[$regionName].Count) lines)"
}

Write-Host "Split $($RegionFileMap.Count) regions from $SourceFile"
Write-Host "Tip: write to a temp directory, then move files and delete the monolith source."

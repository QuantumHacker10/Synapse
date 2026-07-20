param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

function Sanitize-Name([string]$Name) {
    $clean = ($Name -replace '[^A-Za-z0-9]', '')
    if ([string]::IsNullOrWhiteSpace($clean)) { return "Section" }
    return $clean
}

function Split-ByRegions {
    param([string]$SourcePath, [string]$Prefix, [string]$OutDir)

    $lines = Get-Content -Path $SourcePath -Encoding UTF8
    $namespaceIdx = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].Trim().StartsWith("namespace ")) { $namespaceIdx = $i; break }
    }

    $header = $lines[0..$namespaceIdx]
    $body = $lines[($namespaceIdx + 1)..($lines.Count - 2)]
    $regions = @()
    $currentName = $null
    $currentLines = New-Object System.Collections.Generic.List[string]

    foreach ($line in $body) {
        if ($line -match '^\s*#region\s+(.+)') {
            if ($null -ne $currentName) {
                $regions += ,@($currentName, $currentLines.ToArray())
            }
            $currentName = $Matches[1].Trim()
            $currentLines = New-Object System.Collections.Generic.List[string]
            continue
        }
        if ($line -match '^\s*#endregion') {
            if ($null -ne $currentName) {
                $regions += ,@($currentName, $currentLines.ToArray())
            }
            $currentName = $null
            $currentLines = New-Object System.Collections.Generic.List[string]
            continue
        }
        if ($null -ne $currentName) {
            [void]$currentLines.Add($line)
        }
    }

    if ($regions.Count -eq 0) { throw "No regions found in $SourcePath" }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    foreach ($region in $regions) {
        $name = Sanitize-Name $region[0]
        $outPath = Join-Path $OutDir "$Prefix.$name.cs"
        $chunk = @($header + "{" + $region[1] + "}")
        Set-Content -Path $outPath -Value $chunk -Encoding UTF8
    }

    Remove-Item -Path $SourcePath -Force
    Write-Host "Split $(Split-Path $SourcePath -Leaf) -> $($regions.Count) files"
}

function Split-ByTopLevelTypes {
    param([string]$SourcePath, [string]$Prefix, [string]$OutDir, [switch]$FileScopedNamespace)

    $lines = Get-Content -Path $SourcePath -Encoding UTF8
    $namespaceIdx = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i].Trim().StartsWith("namespace ")) { $namespaceIdx = $i; break }
    }

    if ($FileScopedNamespace) {
        $header = $lines[0..$namespaceIdx]
        $body = $lines[($namespaceIdx + 1)..($lines.Count - 1)]
        $footer = @()
    }
    else {
        $header = $lines[0..$namespaceIdx]
        $body = $lines[($namespaceIdx + 1)..($lines.Count - 2)]
        $footer = @("}")
    }

    $typePattern = '^\s*(?:\[.*\]\s*)*(?:public|internal|private|protected)?\s*(?:sealed\s+|static\s+|partial\s+)*(?:class|struct|enum|interface|record)\s+(\w+)'
    $chunks = @()
    $currentName = $null
    $currentLines = New-Object System.Collections.Generic.List[string]
    $depth = 0

    foreach ($line in $body) {
        if ($line -match $typePattern -and $depth -eq 0) {
            if ($null -ne $currentName) {
                $chunks += ,@($currentName, $currentLines.ToArray())
            }
            $currentName = $Matches[1]
            $currentLines = New-Object System.Collections.Generic.List[string]
            [void]$currentLines.Add($line)
            $depth = ($line.ToCharArray() | Where-Object { $_ -eq '{' }).Count - ($line.ToCharArray() | Where-Object { $_ -eq '}' }).Count
            continue
        }

        if ($null -ne $currentName) {
            [void]$currentLines.Add($line)
            $depth += ($line.ToCharArray() | Where-Object { $_ -eq '{' }).Count - ($line.ToCharArray() | Where-Object { $_ -eq '}' }).Count
        }
    }

    if ($null -ne $currentName) {
        $chunks += ,@($currentName, $currentLines.ToArray())
    }

    if ($chunks.Count -eq 0) { throw "No top-level types found in $SourcePath" }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $seen = @{}
    foreach ($chunk in $chunks) {
        $typeName = $chunk[0]
        if (-not $seen.ContainsKey($typeName)) { $seen[$typeName] = 0 }
        $seen[$typeName] = $seen[$typeName] + 1
        $suffix = if ($seen[$typeName] -eq 1) { "" } else { "_$($seen[$typeName])" }
        $outPath = Join-Path $OutDir "$Prefix.$typeName$suffix.cs"
        if ($FileScopedNamespace) {
            $content = @($header + $chunk[1])
        }
        else {
            $content = @($header + "{" + $chunk[1] + $footer)
        }
        Set-Content -Path $outPath -Value $content -Encoding UTF8
    }

    Remove-Item -Path $SourcePath -Force
    Write-Host "Split $(Split-Path $SourcePath -Leaf) -> $($chunks.Count) files"
}

Split-ByRegions (Join-Path $Root "src/Synapse.AI/NeatGEvolutionEngine.cs") "NeatGEvolutionEngine" (Join-Path $Root "src/Synapse.AI")
Split-ByTopLevelTypes (Join-Path $Root "src/Synapse.Rendering/VulkanRhiDevice.cs") "VulkanRhiDevice" (Join-Path $Root "src/Synapse.Rendering")
Split-ByTopLevelTypes (Join-Path $Root "src/Synapse.Physics/Solvers.cs") "Solvers" (Join-Path $Root "src/Synapse.Physics") -FileScopedNamespace

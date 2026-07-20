param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$source = Join-Path $Root "src/Synapse.Rendering/VulkanRhiDevice.cs"
$lines = Get-Content -Path $source -Encoding UTF8

$namespaceIdx = 0
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i].Trim().StartsWith("namespace ")) { $namespaceIdx = $i; break }
}

$prefix = $lines[0..$namespaceIdx]
$bodyStart = $namespaceIdx + 2  # skip namespace line and opening '{'

$chunks = @(
    @{ Name = "Types"; Start = $bodyStart; End = 1392 },
    @{ Name = "Descriptions"; Start = 1393; End = 1890 },
    @{ Name = "Device"; Start = 1891; End = 3997 },
    @{ Name = "Resources"; Start = 3998; End = 5270 },
    @{ Name = "Pipelines"; Start = 5271; End = 5708 },
    @{ Name = "Memory"; Start = 5709; End = 6567 },
    @{ Name = "Sync"; Start = 6568; End = 6893 },
    @{ Name = "Tracking"; Start = 6894; End = 8097 },
    @{ Name = "Layout"; Start = 8098; End = ($lines.Count - 2) }
)

foreach ($chunk in $chunks) {
    $body = $lines[$chunk.Start..$chunk.End]
    $outPath = Join-Path $Root "src/Synapse.Rendering/VulkanRhiDevice.$($chunk.Name).cs"
    $content = @($prefix + "{" + $body + "}")
    Set-Content -Path $outPath -Value $content -Encoding UTF8
    Write-Host "Wrote $($chunk.Name) ($($body.Count) lines)"
}

Remove-Item -Path $source -Force
Write-Host "Removed monolithic VulkanRhiDevice.cs"

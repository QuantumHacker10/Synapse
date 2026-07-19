#Requires -Version 7.0
<#
.SYNOPSIS
  Configure la protection de la branche main sur GitHub.

.DESCRIPTION
  Nécessite gh CLI authentifié (gh auth login).
  Requiert un dépôt public ou GitHub Pro — échoue sur dépôt privé gratuit.

.EXAMPLE
  .\.github\scripts\apply-branch-protection.ps1
  .\.github\scripts\apply-branch-protection.ps1 -Branch develop
#>
param(
    [string]$Owner = "QuantumHacker10",
    [string]$Repo = "Synapse",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $scriptDir "branch-protection.json"

Write-Host "Application de la protection sur ${Owner}/${Repo}:${Branch}..."

try {
    gh api `
        -X PUT `
        "repos/${Owner}/${Repo}/branches/${Branch}/protection" `
        --input $configPath
    Write-Host "Protection appliquee avec succes." -ForegroundColor Green
}
catch {
    Write-Error @"
Echec : $($_.Exception.Message)

Sur un depot prive gratuit, la protection de branche n'est pas disponible via l'API.
Options :
  1. Rendre le depot public (Settings -> General -> Danger Zone)
  2. Passer a GitHub Pro / Team
  3. Configurer manuellement : Settings -> Branches -> Add rule -> pattern '$Branch'
"@
    exit 1
}

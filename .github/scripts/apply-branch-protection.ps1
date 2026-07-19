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

$body = @{
    required_status_checks = @{
        strict   = $true
        contexts = @("test-linux", "publish-windows", "analyze")
    }
    enforce_admins                = $true
    required_pull_request_reviews = @{
        required_approving_review_count = 1
        dismiss_stale_reviews           = $true
        require_code_owner_reviews      = $false
    }
    restrictions                  = $null
    allow_force_pushes            = $false
    allow_deletions               = $false
    required_linear_history       = $false
    required_conversation_resolution = $true
} | ConvertTo-Json -Depth 5

Write-Host "Application de la protection sur ${Owner}/${Repo}:${Branch}..."

try {
    gh api `
        -X PUT `
        "repos/${Owner}/${Repo}/branches/${Branch}/protection" `
        --input - `
        <<< $body
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

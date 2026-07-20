#!/usr/bin/env bash
# Add CODECOV_TOKEN to GitHub Actions secrets (requires gh CLI + repo admin).
set -euo pipefail

REPO="${1:-QuantumHacker10/Synapse}"

echo "Codecov secret setup for $REPO"
echo ""
echo "1. Open: https://app.codecov.io/gh/$REPO/settings"
echo "2. Copy the Repository Upload Token (UUID only)"
echo ""
read -rsp "Paste token (hidden): " TOKEN
echo ""

if [[ -z "$TOKEN" ]]; then
  echo "No token entered." >&2
  exit 1
fi

if ! command -v gh >/dev/null; then
  echo "Install GitHub CLI (gh) or add the secret manually in GitHub Settings → Secrets." >&2
  exit 1
fi

gh secret set CODECOV_TOKEN --repo "$REPO" --body "$TOKEN"
echo "Secret CODECOV_TOKEN set for $REPO."
echo "If OIDC upload still fails, edit build.yml to use token instead of use_oidc (see docs/CODECOV_SETUP.md)."

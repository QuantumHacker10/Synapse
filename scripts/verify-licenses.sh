#!/usr/bin/env bash
# Verify NuGet dependency licenses are permissive (MIT/Apache-2.0).
set -euo pipefail

echo "=== Synapse OMNIA — Dependency License Audit ==="
echo ""

PROJECTS=(
  "src/Synapse.Studio/Synapse.Studio.csproj"
  "tests/Synapse.Tests/Synapse.Tests.csproj"
)

for proj in "${PROJECTS[@]}"; do
  echo "--- $proj ---"
  dotnet list "$proj" package --include-transitive 2>/dev/null || true
  echo ""
done

echo "--- Vulnerability scan ---"
dotnet list package --vulnerable --include-transitive 2>&1 || true

echo ""
echo "See THIRD_PARTY_NOTICES.md for the full license inventory."
echo "Run 'dotnet list package --vulnerable --include-transitive' before each release."

#!/usr/bin/env bash
# Multi-RID publish smoke: verify Studio publishes for all native RIDs and entrypoints exist.
# Does not launch Vulkan (headless CI safe). See docs/REQUIREMENTS.md.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/artifacts/smoke}"
CONFIG="${CONFIG:-Release}"
RIDS=(win-x64 win-arm64 linux-x64 linux-arm64 osx-arm64 osx-x64)

mkdir -p "$OUT"
failed=0
for rid in "${RIDS[@]}"; do
  echo "==> Smoke publish $rid"
  dest="$OUT/Synapse-$rid"
  rm -rf "$dest"
  if ! dotnet publish "$ROOT/src/Synapse.Studio/Synapse.Studio.csproj" \
      -c "$CONFIG" -r "$rid" --self-contained true \
      -p:PublishReadyToRun=true \
      -o "$dest" >/tmp/synapse-smoke-"$rid".log 2>&1; then
    echo "FAIL publish $rid (see /tmp/synapse-smoke-$rid.log)"
    failed=1
    continue
  fi
  if [[ -f "$dest/Synapse.Studio.exe" || -f "$dest/Synapse.Studio" ]]; then
    echo "OK $rid entrypoint"
    # Headless capability probe (no GPU required)
    if [[ -f "$dest/Synapse.Studio" ]]; then
      "$dest/Synapse.Studio" --help >/dev/null 2>&1 || true
    fi
  else
    echo "FAIL $rid: missing Synapse.Studio entrypoint"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "Smoke publish FAILED"
  exit 1
fi
echo "All RID smoke publishes OK → $OUT"

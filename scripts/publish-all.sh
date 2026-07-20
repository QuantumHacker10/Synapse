#!/usr/bin/env bash
# Publish Synapse Studio for the main native RIDs (self-contained).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts"
CONFIG="${CONFIG:-Release}"
RIDS=(win-x64 linux-x64 osx-arm64)

mkdir -p "$OUT"
for rid in "${RIDS[@]}"; do
  echo "==> Publishing $rid"
  dest="$OUT/Synapse-$rid"
  rm -rf "$dest"
  dotnet publish "$ROOT/src/Synapse.Studio/Synapse.Studio.csproj" \
    -c "$CONFIG" \
    -r "$rid" \
    --self-contained true \
    -p:PublishReadyToRun=true \
    -o "$dest"
  cp -f "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/COPYRIGHT" "$dest/" 2>/dev/null || true
  echo "OK $dest"
done

echo "Done. Artifacts under $OUT"

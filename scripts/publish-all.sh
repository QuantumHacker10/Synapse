#!/usr/bin/env bash
# Publish Synapse Studio for native RIDs (self-contained), mid-range multi-platform.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/artifacts"
CONFIG="${CONFIG:-Release}"
# Baseline x64/arm64 on Windows, Linux, macOS — not Windows-x64-only.
RIDS=(win-x64 win-arm64 linux-x64 linux-arm64 osx-arm64 osx-x64)

mkdir -p "$OUT"
for rid in "${RIDS[@]}"; do
  echo "==> Publishing $rid"
  dest="$OUT/Synapse-$rid"
  rm -rf "$dest"
  # ReadyToRun without AVX-512: mid-range CPU baseline (AVX2 / NEON via runtime dispatch).
  dotnet publish "$ROOT/src/Synapse.Studio/Synapse.Studio.csproj" \
    -c "$CONFIG" \
    -r "$rid" \
    --self-contained true \
    -p:PublishReadyToRun=true \
    -o "$dest"
  cp -f "$ROOT/README.md" "$ROOT/LICENSE" "$ROOT/COPYRIGHT" "$ROOT/docs/REQUIREMENTS.md" "$dest/" 2>/dev/null || true
  echo "OK $dest"
done

echo "Done. Artifacts under $OUT"

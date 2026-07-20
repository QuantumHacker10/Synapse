#!/usr/bin/env bash
# Capture a Synapse Studio screenshot (requires Vulkan GPU + display).
# Usage: bash scripts/capture-studio-screenshot.sh [output.png]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/docs/media/synapse-studio-frame.png}"
SCENE="${SYNAPSE_SCENE:-$ROOT/samples/demo.synapse}"

echo "Output: $OUT"
echo "Scene:  $SCENE"

dotnet build "$ROOT/src/Synapse.Studio/Synapse.Studio.csproj" -c Release

if [[ "$(uname -s)" == "Linux" && -z "${DISPLAY:-}" ]]; then
  echo "Starting Xvfb on :99..."
  export DISPLAY=:99
  Xvfb :99 -screen 0 1920x1080x24 &
  sleep 1
fi

echo "Launch Studio in background (close manually or wait for capture)..."
dotnet run --project "$ROOT/src/Synapse.Studio" -c Release -- --scene "$SCENE" &
PID=$!
sleep 8

if command -v scrot >/dev/null; then
  scrot "$OUT"
elif command -v import >/dev/null; then
  import -window root "$OUT"
elif command -v ffmpeg >/dev/null; then
  ffmpeg -y -f x11grab -video_size 1920x1080 -i "$DISPLAY.0" -frames:v 1 "$OUT"
else
  echo "Install scrot, imagemagick, or ffmpeg for screen capture." >&2
  kill "$PID" 2>/dev/null || true
  exit 1
fi

kill "$PID" 2>/dev/null || true
echo "Saved $OUT ($(du -h "$OUT" | cut -f1))"
echo "Regenerate demo: bash scripts/render-demo-media.sh"

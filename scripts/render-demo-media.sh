#!/usr/bin/env bash
# Render README/Release demo media from the studio frame (or a custom PNG).
# Requires: ffmpeg, DejaVu fonts (Linux: fonts-dejavu-core)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MEDIA="$ROOT/docs/media"
FRAME="${1:-$MEDIA/synapse-studio-frame.png}"

if [[ ! -f "$FRAME" ]]; then
  echo "Frame not found: $FRAME" >&2
  exit 1
fi

mkdir -p "$MEDIA"
FONT_BOLD="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if [[ ! -f "$FONT_BOLD" ]]; then
  FONT_BOLD=""
  FONT_REG=""
fi

VF="scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x0a0e14"
VF+=",zoompan=z='min(zoom+0.0008,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=120:s=1280x720:fps=24"
if [[ -n "$FONT_BOLD" ]]; then
  VF+=",drawtext=fontfile=$FONT_BOLD:text='Synapse OMNIA — 3D Simulation Engine':fontsize=28:fontcolor=0x45e0b8:x=(w-text_w)/2:y=40"
  VF+=",drawtext=fontfile=$FONT_REG:text='G-DNN  •  L-DNN  •  Living laws  •  NEAT-G':fontsize=18:fontcolor=white:x=(w-text_w)/2:y=80"
fi

echo "Rendering MP4 → $MEDIA/synapse-demo.mp4"
ffmpeg -y -loop 1 -i "$FRAME" -vf "$VF" -t 5 -c:v libx264 -pix_fmt yuv420p "$MEDIA/synapse-demo.mp4"

echo "Rendering GIF → $MEDIA/synapse-demo.gif"
ffmpeg -y -i "$MEDIA/synapse-demo.mp4" \
  -vf "fps=12,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
  "$MEDIA/synapse-demo.gif"

echo "Done. Replace synapse-studio-frame.png with a real Studio screenshot to refresh."

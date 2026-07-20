# Demo media

Assets for README and GitHub Releases.

| File | Description |
|---|---|
| `synapse-studio-frame.png` | Source frame for demo video (Studio viewport) |
| `synapse-demo.mp4` | 5 s demo video (bundled in releases) |
| `synapse-demo.gif` | Animated preview for README |
| `synapse-studio-frame-vm-no-gpu.png` | Reference capture without Vulkan (black viewport) |

## Refresh demo (GPU machine)

Requires **Vulkan** and a working display (Windows, Linux desktop, or macOS).

```bash
# 1. Capture Studio
bash scripts/capture-studio-screenshot.sh docs/media/synapse-studio-frame.png

# 2. Regenerate GIF + MP4
bash scripts/render-demo-media.sh
```

On **Windows**, take a screenshot manually (Win+Shift+S), save as
`docs/media/synapse-studio-frame.png`, then run `bash scripts/render-demo-media.sh`
(Git Bash or WSL).

## CI / headless VMs

Without a Vulkan GPU, Studio opens but the 3D viewport stays black (`ErrorIncompatibleDriver`).
The committed `synapse-studio-frame.png` uses a rendered mockup until a maintainer replaces it
from a machine with a GPU.

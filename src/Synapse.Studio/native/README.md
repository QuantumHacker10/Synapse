# Native dependencies

## Windows x64
Place `glfw3.dll` (GLFW 3.4+) in this folder before publishing — it is copied next to `Synapse.Studio.exe`.
Vulkan runtime comes from the GPU driver (or LunarG Vulkan SDK).

## Linux
Install system packages, for example:
```bash
# Debian/Ubuntu
sudo apt install libglfw3 libvulkan1
```
The loader resolves `libglfw.so.3` and `libvulkan.so.1` automatically.
Surface extensions come from `glfwGetRequiredInstanceExtensions` (X11 or Wayland).

Optional: ship natives under `runtimes/linux-x64/native/` for self-contained layouts.

## macOS
Install GLFW (Homebrew: `brew install glfw`) and MoltenVK / Vulkan SDK.
The loader looks for `libglfw.3.dylib` and `libvulkan.1.dylib` (or `libMoltenVK.dylib`).
Instance creation enables `VK_KHR_portability_enumeration` for MoltenVK.

Optional: ship natives under `runtimes/osx-arm64/native/` for self-contained layouts.

## Multi-RID publish
```bash
bash scripts/publish-all.sh
```
Produces `artifacts/Synapse-{win-x64,linux-x64,osx-arm64}/`. CI release tags (`v*`) build the same matrix.

## Studio viewport
- **Windows**: Avalonia can embed Vulkan via a child HWND (`VK_KHR_win32_surface`).
- **Linux / macOS**: Studio opens a GLFW sibling window for rendering (HWND embedding is Windows-only).

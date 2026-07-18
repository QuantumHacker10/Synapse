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

## macOS
Install GLFW (Homebrew: `brew install glfw`) and MoltenVK / Vulkan SDK.
The loader looks for `libglfw.3.dylib` and `libvulkan.1.dylib` (or `libMoltenVK.dylib`).
Instance creation enables `VK_KHR_portability_enumeration` for MoltenVK.

## Studio viewport
- **Windows**: Avalonia can embed Vulkan via a child HWND (`VK_KHR_win32_surface`).
- **Linux / macOS**: Studio opens a GLFW sibling window for rendering (HWND embedding is Windows-only).

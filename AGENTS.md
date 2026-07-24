# AGENTS.md

## Cursor Cloud specific instructions

Synapse OMNIA is a **.NET 10** (C# 14) 3D-simulation engine. Standard build/test/run
commands live in `README.md` and `CONTRIBUTING.md`; only the non-obvious, cloud/headless
caveats are captured here.

### Toolchain
- .NET SDK **10.0.300** (pinned by `global.json`) is installed at `/usr/local/dotnet` and
  symlinked to `/usr/local/bin/dotnet` (on `PATH`). The startup update script runs
  `dotnet restore Synapse.slnx`.

### Services / how to work with them
The repo is a single product built from one solution (`Synapse.slnx`): ten libraries under
`src/` plus the **Synapse Studio** Avalonia desktop editor (`src/Synapse.Studio`) and one
xUnit test project (`tests/Synapse.Tests`).

- Build: `dotnet build Synapse.slnx -c Release` (0 errors; ~2000 analyzer warnings are expected).
- Test: `dotnet test Synapse.slnx --no-build -c Release` (suite xUnit complète).
- Health: `dotnet run --project src/Synapse.Studio -c Release -- --health`
- Lint (matches CI `analysis.yml`): `dotnet format whitespace --verify-no-changes`.
- Run the editor: `DISPLAY=:1 dotnet run --project src/Synapse.Studio -c Release`.

### Headless / GPU caveats (important, non-obvious)
- The official target is **Windows x64 + a real Vulkan GPU**. This cloud VM is headless Linux
  with only a software Vulkan (llvmpipe/lavapipe) renderer.
- The Studio app P/Invokes the Vulkan loader under its Windows name `vulkan-1.dll`. On Linux
  it does not resolve that name, so the embedded 3D viewport fails to start with a **caught**
  `DllNotFoundException` (logged `[Viewport] Failed to start render engine`) and the editor
  UI stays fully usable. This is the expected, graceful headless path — do not try to "fix" it
  by symlinking `libvulkan.so.1` to `libvulkan-1.dll.so`: doing so lets the renderer partially
  init on lavapipe and then **crashes the whole process natively** (`malloc_consolidate: invalid
  chunk size`). `libglfw3` is installed and resolves fine; the Vulkan loader is intentionally
  left unresolvable under the Windows name.
- Because the per-frame `FrameOrchestrator` tick is driven by the viewport's render timer, the
  live status-bar counters (IPS, Sim ms, `entités=`, Physique) stay at 0 on this headless VM
  even while the simulation is "playing". Editor actions (create/select/inspect entities,
  compile living laws, LLM console, blueprint editing) work normally; only real-time rendered
  stepping needs a GPU.
- The engine-only mode (`dotnet run --project src/Synapse.Studio -- --engine`) also requires a
  working Vulkan device and will not run on this headless VM.

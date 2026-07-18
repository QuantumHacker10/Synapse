# Changelog

## 1.1.0 — 2026-07-18

### Added
- Cross-platform Vulkan via GLFW (`libglfw` / MoltenVK) ; HWND embed reste Windows-only
- `DiffScalar2` (AD ordre 2) + `StochasticFieldDifferentiator.ComputeGradient` (API Span)
- Blueprint editor (graph JSON, compile → agent), sculpt strokes, Megascans import UI
- `TreatWarningsAsErrors` sur Runtime/Studio ; suppression ciblée du bruit analyseurs legacy

## 1.0.0 — 2026-07-18

### Added
- **G-DNN Studio** Avalonia (viewport, scène, lois, évolution, LLM, simulation, performances)
- **Synapse.Runtime** — `EngineHost`, `FrameOrchestrator`, `SceneDocument` (projets `.synapse`)
- Logging fichier/console et `appsettings.json` / CLI / variables d’environnement
- Surface Vulkan Win32 pour embarquement Avalonia + API caméra / pause sur `RenderEngine`
- Streaming d’assets G-DNN depuis le dossier `assets/` du projet
- Jumeaux numériques in-memory
- Coverlet, CI Windows publish, workflow GitHub Release
- Mode `Synapse.Studio.exe --engine` pour la boucle GLFW seule

### Fixed
- Format Guid invalide (`:N8`) dans `SentientEntityFactory.CreateNPC`

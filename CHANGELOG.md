# Changelog

All notable changes to **Synapse OMNIA** are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.3.0] — 2026-07-20

Release **Synapse OMNIA 1.3** — MIT license, community documentation, demo media, Codecov CI.

### Added

- English README, CONTRIBUTING, and GitHub issue/PR templates for international contributors.
- Documentation: [docs/getting-started.md](docs/getting-started.md), [docs/architecture.md](docs/architecture.md), [docs/LICENSE-OPTIONS.md](docs/LICENSE-OPTIONS.md), [docs/PROMOTION.md](docs/PROMOTION.md).
- Demo media for README and Releases: [docs/media/synapse-demo.gif](docs/media/synapse-demo.gif), [docs/media/synapse-demo.mp4](docs/media/synapse-demo.mp4), [scripts/render-demo-media.sh](scripts/render-demo-media.sh).
- Codecov integration (`codecov.yml`) and dynamic coverage badge in README.

### Changed

- **License switched to MIT** — replaces proprietary license; forks and contributions welcome.
- Release workflow bundles demo media; release notes reference MIT License.
- CI uploads Cobertura coverage to Codecov via OIDC; see [docs/CODECOV_SETUP.md](docs/CODECOV_SETUP.md).
- Product version **1.3.0** (`Directory.Build.props`).

## [1.2.0] — 2026-07-20

Release industrielle **Synapse OMNIA 1.2** : physique/rendu durcis, multiplateforme native, joints/véhicules/mesh, Studio Omnia, publish multi-RID.

### Ajouté

- **Synapse Studio** : viewport Vulkan embarqué (HWND), grille/gizmos, outils sélection/déplacement/rotation, édition d'entités et blueprint graphique.
- Inspecteur Omnia : import mesh (glTF/OBJ), véhicule raycast, bake G-DNN SDF, charnière monde, joint distance soft, resync physique.
- Lecture G-buffer GPU pour l'illumination L-DNN (fallback constantes si indisponible).
- **Physique industrielle** : `RigidBodyWorld` (primitives, AABB, contacts analytiques, PGS, sleep) et `MultiphysicsOrchestrator` (pas fixe, living laws + rigid bodies + continuum optionnel), branchés dans `EngineHost.TickPhysics`.
- **Contacts mesh précis** : sphère ↔ triangle mesh / convex hull (point le plus proche sur triangle).
- **Soft constraints** : `PhysicsJoint.Compliance` (CFM) sur distance, point-to-point, slider.
- **Rendu industriel** : SSAO réel (`LDNNBridge.ComputeAO`), kernels compute CPU (`ssao` / `blur_ao` / `downsample_irradiance`), simplification mesh QEM (`QuadricMeshSimplifier`).
- **Compilation DXC native** : `ShaderCompiler` appelle DXC/glslang via `SpirvToolchain` (SPIR-V), fallback simulé explicite et journalisé (`ShaderCompilationBackend`).
- **GI GPU-résidente** : `GpuResidentGiPipeline` + kernel `ssgi_irradiance` — plus de fill constant tant qu'un G-buffer GPU est résident.
- **CCD** : détection continue (TOI) sphère/plan et sphère/AABB pour empêcher le tunneling.
- **Certification CFD/FEA** : `PhysicsCertification.RunIndustrialCore()` (repos, moment, CCD, onde, CFL Maxwell, élasticité, SPH).
- **Multiplateforme native** : `NativePlatform` + `IVulkanSurfaceFactory` (GLFW primaire partout, HWND optionnel Windows), résolution native élargie (`runtimes/{rid}/native`).
- **Joints avancés** : hinge, ball-socket, slider, fixed, distance (PGS) — persistés dans la scène (`SceneJointData`).
- **Véhicules** : `VehicleController` raycast (suspension, steer/drive/brake).
- **SynapseMeshProvider** : mesh → cook convex/triangle → corps + bake G-DNN SDF optionnel.
- CI : publish linux-x64, job certification industrielle, release matrix win/linux/osx.
- Script `scripts/publish-all.sh` pour les trois RID principaux.
- Tests de validation industrielle (repos, moment, SSAO, LOD, tick runtime, joints, mesh, soft constraints).

### Modifié

- Interface Studio entièrement en français (menus, statuts, dialogues, console LLM).
- Texte « À propos » enrichi (onglet Performance) aligné sur le README et le site vitrine (v1.2).
- README, site GitHub Pages et CHANGELOG recentrés sur le positionnement « moteur de simulation 3D ».
- Remplacement de la licence MIT par une **licence propriétaire** (anti-copie, anti-fork, anti-plagiat) — voir [LICENSE](LICENSE) et [COPYRIGHT](COPYRIGHT).
- Remplacement du LOD stochastique placeholder et des stubs AO / compute no-op.
- `SceneRenderer.RenderGI` réutilise le G-buffer résident au lieu de forcer des constantes chaque frame.
- Version produit **1.2.0** (`Directory.Build.props`).

### Technique

- **.NET 10** / C# 14, Vulkan (Windows HWND + GLFW, Linux, macOS MoltenVK).
- Tag Git prévu : `v1.2.0`.

---

## [1.1.0] — 2026-07-19

Première release produit **Synapse OMNIA 1.1** : Synapse Studio, moteur de simulation 3D unifié (physique, évolution, habitants, rendu temps réel).

### Ajouté

- Import initial du moteur SYNAPSE OMNIA 1.1 : projets Core, Physics, AI, Genomics, Rendering, LLM, Simulation, Infrastructure, Runtime et **Synapse Studio**.
- Site vitrine (`site/`) et déploiement GitHub Pages.
- Pipeline CI : build & tests, analyse statique, release automatique sur tag `v*`.
- **G-DNN** : SDF neuronaux, polygonisation LOD, meshlets, export glTF.
- **L-DNN** : illumination neuronale, GI hybride, ombres et reflets neuronaux.
- **Living laws** : 100 lois physiques textuelles sur 18 domaines, hot-reload.
- **NEAT-G** : évolution de formes, sélection NSGA-II.
- **HybridLlmRouter** : ONNX, Ollama, OpenAI, Anthropic, Gemini, Azure.
- Agents sentients : perception, behavior trees, ordonnanceur actif/dormant.
- Documentation README alignée sur l'état réel (CLI, Pages, Azure, 100 lois).

### Modifié

- Refactorisation des monolithes LDNN et LLM ; APIs d'intégration documentées.
- Suppression de code mort et consolidation des lots G-DNN / L-DNN.

### Technique

- **.NET 10** / C# 14, Vulkan (Windows HWND + GLFW, Linux, macOS MoltenVK).
- Tag Git : [`v1.1.0`](https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0).

[1.3.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.3.0
[1.2.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.2.0
[1.1.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0

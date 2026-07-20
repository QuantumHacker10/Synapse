# Changelog

Toutes les modifications notables de **Synapse OMNIA** sont documentées ici.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté

- **GETTING_STARTED.md** — guide de démarrage rapide avec exemples concrets (CLI, scènes `.synapse`, intégration C#).
- **docs/API.md** — référence des APIs publiques par module.
- **docs/ARCHITECTURE.md** — diagrammes Mermaid (pipeline, modules, CI/CD).
- **docs/screenshots/** — maquettes SVG Studio (vue principale, rendu G-DNN/L-DNN).
- **ROADMAP.md**, **COMMUNITY.md**, **CODE_OF_CONDUCT.md**, **SECURITY.md**.
- **NOTICE** — avis standard (crédits, marques).
- **THIRD_PARTY_NOTICES.md** — inventaire des licences des dépendances NuGet et natives.
- **LICENSE_HISTORY.md** — historique des changements de licence (MIT → propriétaire → MIT).
- **13 tests unitaires** pour `PhysicsState` (`PhysicsStateStructTests.cs`).
- Templates GitHub : issues (bug, feature), discussions (Q&A, idées).
- CI : workflow **CodeQL**, **Dependabot**, job **test-macos**, upload **Codecov**, audit NuGet.
- `coverlet.runsettings`, `codecov.yml`, `scripts/verify-licenses.sh`.
- Badges README : Analysis, CodeQL, Codecov, Tests.

### Modifié

- README enrichi : exemples C#, captures d'écran, liens communauté, badges CI étendus.
- **PhysicsState.cs** (276 Ko) découpé en **26 fichiers** modulaires sous `Synapse.Core/` (`Vector3D.cs`, `PhysicsStateCore.cs`, `UnitConverter.cs`, etc.).

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

- Passage à la **licence MIT** (open source) — remplace la licence propriétaire ; voir [LICENSE](LICENSE) et [COPYRIGHT](COPYRIGHT).
- Interface Studio entièrement en français (menus, statuts, dialogues, console LLM).
- Texte « À propos » enrichi (onglet Performance) aligné sur le README et le site vitrine (v1.2).
- README, site GitHub Pages et CHANGELOG recentrés sur le positionnement « moteur de simulation 3D ».
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

[1.2.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.2.0
[1.1.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0

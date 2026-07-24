# Changelog

Toutes les modifications notables de **Synapse OMNIA** sont documentées ici.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté

- **Tests de couverture** — CLI/env config, ScenePatchCodec, STUN/NAT, validation SceneDocument, collaboration EngineHost/orchestrator, VR fail-closed, PeerEncryption, WasmStudioPublisher, plugins/sentience/infra
- **OpenXR natif** — `NativeOpenXrRuntime` (Silk.NET), session réelle + fallback `SYNAPSE_VR_SIMULATE=1`
- **NAT réel** — client STUN RFC 5389, rendez-vous UDP, hole-punch, hub WAN AES-GCM
- **Studio web WASM** — projet `Synapse.Web.Studio` (Blazor), `--export-web` / `WasmStudioPublisher`
- **Matrice de maturité** — [`docs/MATURITY.md`](docs/MATURITY.md), `FeatureMaturityCatalog`, attribut `[SynapseExperimental]`
- **Scène lab** — `samples/lab-heat-agents.synapse` (loi heat, agents, joint hinge) + tests de chargement CI
- **Plugin trust** — `PluginTrustMode.RequireManifest` + `plugin.synapse.json` (SHA-256), env `SYNAPSE_PLUGIN_TRUST`
- **Tests de durcissement** — mmap, marketplace, VR, WAN, GPU upload, scènes/blueprints hostiles
- **Budgets blueprint** — spawns / LLM / nœuds par tick ; arrêt sur boucle / edge manquante
- **Runtime diagnostics** — `LastRuntimeError`, `RuntimeErrorCount`, `IsLawDegraded`

### Modifié

- **VR / WAN / Web** : features de première classe branchées sur `EngineHost`, `FrameOrchestrator` et Studio (menus Collab, status bar, patches scène)
- **VR / WAN / Web** promus **EarlyAccess** (chemins natifs branchés ; pas encore Supported)
- Positionnement honnête **accès anticipé / R&D** (README, ROADMAP, site, SECURITY)
- **GLB** : chunks bornés + plafonds fichier / data URI
- **ZeroCopyBuffer / MappedBuffer** : mmap réel, pin handle, `checked`, `createIfMissing`
- **AssetStreamer / GpuUpload** : fail-closed + hash d'intégrité
- **LawMarketplace / Plugins / P2P / WAN / VR** : trust, framing, registre, simulate gate
- **SceneDocument / Blueprint** : validation structurelle ; JSON corrompu = exception
- **FrameOrchestrator** : ticks non réentrants
- **LLM** : Gemini `x-goog-api-key`, OpenAI retry/dispose/cap, `MaxLatencyMs`, sessions path-safe
- **Studio** : erreurs projet / blueprint / Megascans / LLM / viewport
- **SSRF** : DNS optionnel (activé sur downloads) ; PostProcess borné
## [2.10.0] — 2026-07-21

**OpenUSD MeshIO production-complete** — topology DCC, normals, multi-mesh, purpose, health smoke.

### Ajouté

- **`faceVertexCounts`** triangulation (+ sentinels / flat tris)
- **Authored normals**, multi-`def Mesh`, `purpose` / `visibility`, `extent`, `doubleSided`
- **`UsdMeshTopology`**, **`UsdProductionSmoke`**, health `usd=ok` / `UsdRuntimeReady`
- Material `opacityThreshold`, `emissiveIntensity`, `sourceColorSpace`, UDIM multi-slot
- Composition **payload mute** (`UsdIncludePayloads`), prefer `bindTransforms`, stage FPS
- BlendShape **normalOffsets** Apply; sample `production_dcc.usda`

### Modifié

- Version **2.10.0**, `docs/PRODUCTION.md` honesty (MeshIO first-party, pas Hydra)

## [2.9.0] — 2026-07-21

**Runtime USD étendu** — UDIM/MDL, blend shapes, streaming GPU textures, marketplace plugins distant.

### Ajouté

- **UDIM** — `UsdUdim.ExpandTiles` / `MeshMaterial.UdimTiles` (`<UDIM>` / `%(UDIM)`)
- **MDL** — `MdlAssetPath` / `MdlMaterialName` (`sourceAsset`, `mdl:sourceMaterial`)
- **UsdSkel BlendShape** — `MeshBlendShape` + offsets / Apply
- **`GpuTextureStreamer`** — page-in disque/HTTPS, LRU, prefetch UDIM
- **`RemotePluginMarketplace`** — catalog HTTPS + download hash-vérifié + jail
- CLI `--plugin-marketplace-url` / `SYNAPSE_PLUGIN_MARKETPLACE_URL`
- Sample `udim_mdl_blend.usda`

### Modifié

- Version **2.9.0**, docs honesty : runtime Synapse (pas lib OpenUSD C++ native)

## [2.8.0] — 2026-07-21

**UsdSkel animation clips** — `SkelAnimation` timeSamples + évaluation TRS.

### Ajouté

- **`MeshAnimationClip` / `MeshJointCurve`** sur `MeshAsset`
- **`UsdSkelAnimationParser`** — `def SkelAnimation`, translations/rotations/scales (static + `.timeSamples`), `skel:animationSource`
- Échantillonnage linéaire / slerp (`Evaluate` / `EvaluateLocalMatrices`)
- Sample `samples/meshes/skel_anim_wave.usda`

### Modifié

- Version **2.8.0**, docs production / roadmap

## [2.7.0] — 2026-07-21

**OpenUSD textures avancées** — cartes PBR UsdUVTexture + UVs `primvars:st`.

### Ajouté

- **UsdUVTexture** connectées à UsdPreviewSurface (`diffuseColor`, `normal`, metallic/roughness/occlusion, emissive, displacement)
- Slots `MeshMaterial` : clearcoat / specular / opacity + chemins résolus relatifs au USDA
- UVs `texCoord2f[]` / `float2[] primvars:st`
- Sample `samples/meshes/textured_pbr.usda` (+ placeholders `textures/`)

### Modifié

- Version **2.7.0**, docs production / roadmap

## [2.6.0] — 2026-07-21

**STUN/TURN + OpenUSD DCC complet** — NAT symétrique, materials / skeletons / variants.

### Ajouté

- **STUN** (`StunClient`, RFC 5389 Binding + XOR-MAPPED-ADDRESS)
- **TURN** (`TurnClient`, Allocate / CreatePermission / ChannelBind / ChannelData)
- **`NatIceAssist`** + CLI `--stun-server` / `--turn-server` / `--turn-user` / `--turn-password` / `--wan-prefer-turn`
- **OpenUSD materials** — UsdPreviewSurface + `material:binding`
- **OpenUSD skeletons** — joints, bindTransforms, skel jointIndices/Weights → `MeshSkeleton` / bone attrs
- **OpenUSD variants** — `variantSet` + `MeshLoadConfig.UsdVariantSelections`
- Samples `tetra_preview_skel.usda`, `variant_modeling.usda`

### Modifié

- Version **2.6.0**, `docs/PRODUCTION.md` matrice 2.6
- WAN REGISTER/PEER portent `mode` (`tcp|stun|turn`) et IP STUN optionnelle

## [2.5.0] — 2026-07-21

**Production-ready VR / WAN / OpenUSD DCC** — OpenXR natif+simulé, NAT hors-loopback, xform stacks complets.

### Ajouté

- **OpenXR Vulkan2 natif** — `OpenXrNative` + session/swapchain réels via `XR_KHR_vulkan_enable2` ; fallback simulé labellisé (`IsSimulated`)
- **WAN NAT** — rendezvous UDP sur `IPAddress.Any`, `PEER|session|ip|port`, CLI `--wan-rendezvous` / `--wan-rendezvous-port` / `--wan-join`
- **OpenUSD DCC** — `UsdXform` (translate / rotateXYZ / scale / transform + xformOpOrder), inherits, cibles `@file@</Prim>`, matrices parent×local
- Sample `samples/meshes/composed_xform.usda`

### Modifié

- Version **2.5.0**, `docs/PRODUCTION.md` (matrice 2.5)
- Health notes alignées sur surfaces désormais production

## [2.4.0] — 2026-07-21

**Production-ready desktop** — dispose sûr, health check, docs alignées, plugins Studio, marketplace local, honesty VR/WAN.

### Ajouté

- **`docs/PRODUCTION.md`** — matrice production vs expérimental + checklist release
- **`--health`** — `ProductionHealthReport` (core-ready / interactive-ready)
- **Plugin marketplace local** — `marketplace.json` + vérification SHA-256
- **USD `xformOp:translate`** appliqué à l’import USDA
- Studio Bootstrap charge `--plugin-dir` (parité moteur)

### Sécurité / robustesse

- `EngineHost.DisposeAsync` idempotent + try/catch par ressource
- `PeerConnection` dispose `TcpClient` (plus de fuite socket)
- `PluginHost` protege `ALC.Unload`
- Screenshot capture dispose host/logger dans `finally`
- `--wan-code` câblé (hub authentifié loopback QA)
- SECURITY.md / CONTRIBUTING.md / badges docs synchronisés sur 2.4

### Modifié

- Version **2.4.0**, Codecov **70 %**
- OpenXR documenté comme swapchain simulé (expérimental)

## [2.3.1] — 2026-07-21

Durcissement **production-ready early** : QA multi-RID, composition USD, couverture 60 %, sécurité plugins/P2P.

### Ajouté

- **CI publish-smoke** — matrice 6 RID (`build.yml`) + `scripts/smoke-publish-rids.sh`
- **USD composition arcs** — `UsdCompositionResolver` (references / payloads / subLayers), samples `composed_root.usda` / `composed_usdc.usda`
- **Sécurité plugins** — path jail, blocage UNC/URL, `plugins.allow` SHA-256, deps limitées au dossier plugin
- **Sécurité P2P** — salt par session, AAD AES-GCM, handshake HMAC, `publicBind` exige auth, decrypt-or-drop WAN, MaxPeers, `ReadExactly` framing
- **Tests** — LivingLawCompiler API, LawRegistry lifecycle, RigidBody lifecycle, EngineHost scène, production hardening

### Modifié

- Codecov projet **60 %** (patch 40 %)
- Roadmap v2.3 items prod cochés

## [2.3.0] — 2026-07-21

Release **Synapse OMNIA 2.3** : multi-plateforme milieu de gamme, USDC, blueprints live, couverture ~50 %.

### Ajouté

- **Multi-plateforme natif** — RID `win-x64|win-arm64|linux-x64|linux-arm64|osx-arm64|osx-x64` ; Vulkan loader via `NativeLibraryResolver` (`vulkan-1` → `.dll`/`.so`/`.dylib`)
- **Baseline mid-range** — Vulkan API **1.2**, extensions optionnelles filtrées, scoring GPU (discret > iGPU), features conservatrices
- **SIMD** — `CpuCapabilityProbe` (AVX2/NEON par défaut ; AVX-512 opt-in `SYNAPSE_ALLOW_AVX512`)
- **USDC** — `UsdBinaryLoader` (mesh-pack Synapse + extraction best-effort), sample `samples/meshes/tetra.usdc`
- **Blueprints live** — `EngineHost.HotReloadBlueprint`, exécuteur branché sur le tick sim, édition Studio (Ouvrir/Enregistrer/live checkbox)
- **docs/REQUIREMENTS.md** — configuration minimale matérielle / logicielle
- **Tests v2.3** — plateforme, USDC, blueprints, plugins, lois ; Codecov cible **50 %**

### Modifié

- Version produit **2.3.0**
- `scripts/publish-all.sh` et `release.yml` : 6 RID
- Roadmap : items USDC / blueprints live / couverture 50 % cochés

## [2.2.0] — 2026-07-20

Release **Synapse OMNIA 2.2** : captures Studio live, scaffolds P2P WAN / OpenXR / éditeur web (voir maturité).

### Ajouté

- **Captures PNG live Studio** — `--screenshot`, Avalonia Headless + Skia (`StudioScreenshotCapture`)
- **OpenXR swapchain Vulkan** — `OpenXrVulkanSwapchain` (acquire/release/submit), intégré à `VrSession`
- **P2P WAN** — `NatTraversalCoordinator` (UDP rendezvous), `WanSimulationPeerHub` (AES-GCM via PBKDF2)
- **Éditeur web glTF interactif** — `site/editor/` (WebGPU, hiérarchie, inspecteur), `demo.gltf`
- **5 tests v2.2** — chiffrement, swapchain, WAN, web, VR (266 tests au total)

### Modifié

- Version produit **2.2.0** (`Directory.Build.props`)
- CLI : `--screenshot`, `--wan-session`, `--wan-port`
- `MultiPeerSimulationHub.StartHostAsync(publicBind: true)` pour bind WAN

## [2.1.0] — 2026-07-20

### Ajouté (v2.1)

- **Captures PNG Studio** — `docs/screenshots/*.png`, script `generate-studio-screenshots.py`
- **Découpage monolithes** — `LivingLawLibrary` (12 fichiers), `EntityBehaviorSystem` (36 fichiers)
- **P2P multi-pairs** — `MultiPeerSimulationHub` (TCP localhost)
- **OpenXR production** — `OpenXrVulkanSession` avec détection loader
- **Éditeur web WASM/WebGPU** — `WebEditorBuilder`, site `site/editor/`

## [2.0.0] — 2026-07-20

Release **Synapse OMNIA 2.0** : maturité produit, écosystème extensible et réduction des freins à l'adoption.

### Ajouté

- **API plugins C#** (`Synapse.Plugins`) — chargement sandboxé via `AssemblyLoadContext`, flag `--plugin-dir`
- **Benchmarks headless** — `BenchmarkRunner`, `--benchmark`, suite `samples/benchmarks/default.json`
- **Reproductibilité** — `SimulationSeed`, variable `SYNAPSE_SEED`, flag `--seed`
- **Import FBX ASCII + USD USDA** — `FbxAsciiLoader`, `UsdAsciiLoader`
- **Export scène glTF** — `SceneGlTFExporter`, flag `--export-scene`
- **Blueprint runtime** — `BlueprintRuntimeExecutor` (exécution graphe en simulation)
- **Marketplace lois** — format `.synapse-law`, `LawMarketplace`
- **Fondations v2** — P2P local (`Synapse.Network`), VR stub OpenXR (`Synapse.VR`), preview web (`Synapse.Web`)
- **Découpage `LivingLawCompiler`** — 68 fichiers modulaires (script `split-monoliths.py` corrigé)
- **10 tests v2** — couverture Codecov portée à **40 %**

### Modifié

- Version produit **2.0.0** (`Directory.Build.props`)
- CLI enrichi : `--benchmark`, `--benchmark-out`, `--export-scene`, `--plugin-dir`, `--seed`
- Seuil Codecov projet : **40 %** (patch 30 %)

## [1.3.0] — 2026-07-20

Release **Synapse OMNIA 1.3** : robustesse, découpage des monolithes, analyseurs et inspecteur live Studio.

### Ajouté

- **Mode inspecteur live** dans Synapse Studio (onglet Live) : flux NEAT-G et living laws en temps réel.
- **Tests de régression** NEAT-G (`NeatGEvolutionEngineTests`) et Vulkan (`VulkanRhiDeviceTests`).
- **Scripts** `scripts/split-monoliths.py` pour découper les gros fichiers C# par `#region` ou type top-level.
- **GETTING_STARTED.md**, **docs/API.md**, **docs/ARCHITECTURE.md**, **docs/screenshots/** (maquettes SVG Studio).
- **ROADMAP.md**, **COMMUNITY.md**, **CODE_OF_CONDUCT.md**, **SECURITY.md**.
- CI : workflow **CodeQL**, **Dependabot**, job **test-macos**, upload **Codecov**, audit NuGet.
- `SynapseProduct` — version produit centralisée depuis `Directory.Build.props`.

### Modifié

- **Logs structurés** : remplacement des `catch { }` silencieux par `SynapseLogger` (AI, Rendering, LLM, Simulation, Physics, Genomics).
- **Découpage monolithes** : `NeatGEvolutionEngine` (65 fichiers), `VulkanRhiDevice` (9), `Solvers` (56).
- **Analyseurs** : CA1062/CA2007 réactivés sur `Synapse.Core` et `Synapse.Runtime`.
- **`LivingLawCompiler`** : `LawEventSystem` câblé (compilation, hot-reload, version tree).
- **`EngineHost`** : événements inspecteur (`InspectorFeedEntryAdded`), évolution live par génération.
- Documentation, README, site vitrine et Studio alignés sur **v1.3** et **248 tests**.

### Technique

- Version produit **1.3.0** (`Directory.Build.props`).
- Tag Git : `v1.3.0`.

---

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

[1.3.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.3.0
[1.2.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.2.0
[1.1.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0

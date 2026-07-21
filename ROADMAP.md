# Roadmap — Synapse OMNIA

Vision publique du projet. Les dates sont indicatives ; l'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).


**Maturité :** v2.10 = production desktop (OpenUSD MeshIO, STUN/TURN, OpenXR) — voir [docs/PRODUCTION.md](docs/PRODUCTION.md) et [docs/MATURITY.md](docs/MATURITY.md).

## v2.10 — Actuel (2026-07) — OpenUSD MeshIO production-complete

- [x] `faceVertexCounts` + authored normals + multi-mesh
- [x] purpose / visibility / extent / doubleSided
- [x] payload mute, stage timeCodesPerSecond, bindTransforms preference
- [x] `UsdProductionSmoke` + `--health` `usd=ok`
- [x] Sample `production_dcc.usda`

## v2.9 — Livré (2026-07) — UDIM/MDL, blend shapes, texture stream, marketplace distant

- [x] UDIM tiles + MDL material references
- [x] UsdSkel BlendShape morph targets
- [x] `GpuTextureStreamer` (LRU / prefetch)
- [x] `RemotePluginMarketplace` (catalog + install hash-vérifié)
- [x] Sample `udim_mdl_blend.usda`

## v2.8 — Livré (2026-07) — UsdSkel animation clips

- [x] `SkelAnimation` (translations / rotations / scales + timeSamples)
- [x] `skel:animationSource` + évaluation TRS sur `MeshAnimationClip`
- [x] Sample `skel_anim_wave.usda`

## v2.7 — Livré (2026-07) — OpenUSD textures PBR

- [x] UsdUVTexture → albedo / normal / ORM / emissive / AO / height (+ clearcoat/specular/opacity)
- [x] UVs `primvars:st` sur mesh USDA
- [x] Sample `textured_pbr.usda`

## v2.6 — Livré (2026-07) — STUN/TURN + OpenUSD DCC

- [x] STUN Binding + TURN Allocate/Permission/ChannelData
- [x] OpenUSD materials (UsdPreviewSurface) / skeletons / variantSets
- [x] `docs/PRODUCTION.md` 2.6

## v2.5 — Livré (2026-07) — production VR / WAN / OpenUSD

- [x] OpenXR Vulkan2 natif + fallback simulé production (`IsSimulated`)
- [x] WAN NAT hors-loopback (UDP Any + IP:port + CLI join/rendezvous)
- [x] OpenUSD DCC : composition + xformOp TRS/transform + inherits / prim-path
- [x] `docs/PRODUCTION.md` 2.5

## v2.4 — Livré (2026-07) — production-ready desktop

- [x] Checklist production (`docs/PRODUCTION.md`) + `--health`
- [x] Dispose / fuites / plugins Studio / WAN CLI câblé
- [x] Marketplace plugins local (`marketplace.json`)
- [x] USD translate xformOps + composition arcs
- [x] Couverture Codecov **70 %**
- [x] Docs sécurité / contribution / honesty VR·WAN

## v2.3 — Livré (2026-07)

- [x] Multi-plateforme natif élargi (6 RID, Vulkan 1.2 mid-range, AVX2/NEON baseline)
- [x] Configuration minimale documentée (`docs/REQUIREMENTS.md`)
- [x] Import USD binaire (USDC) — mesh-pack Synapse + best-effort OpenUSD
- [x] Blueprint éditable temps réel dans Studio (hot-reload live)
- [x] Couverture de code cible **50 %** (Codecov)
- [x] QA publish smoke **6 RID** (CI `publish-smoke`)
- [x] Import USDC/USDA **composition arcs** (references / payloads / subLayers)
- [x] Couverture cible **60 %** Physics/Runtime (Codecov)
- [x] Durcissement sécurité plugins (path jail + allowlist) et P2P (auth HMAC + decrypt-or-drop)

## v2.0 — Livré (2026-07)

- [x] API plugin C# (chargement DLL sandboxé)
- [x] Mode headless benchmarks automatisés
- [x] Reproductibilité simulation (`SYNAPSE_SEED`)
- [x] Import FBX ASCII + USD USDA
- [x] Export scène glTF (entités + métadonnées Synapse)
- [x] Blueprint runtime (exécution graphe en simulation)
- [x] Marketplace lois (`.synapse-law`)
- [x] Découpage `LivingLawCompiler.cs` (68 fichiers)
- [x] Couverture de code > 40 % (seuil CI Codecov)
- [x] Fondations P2P, VR (OpenXR stub), preview web
- [x] Tutoriels écrits + guide benchmarks (docs/)

## v2.1 — Livré (2026-07)

- [x] Captures PNG Studio dans README (`docs/screenshots/*.png`)
- [x] Découpage `LivingLawLibrary.cs` et `EntityBehaviorSystem.cs`
- [x] P2P multi-pairs (`MultiPeerSimulationHub`)
- [x] OpenXR session production (`OpenXrVulkanSession`)
- [x] Éditeur web WebGPU (`site/editor/`, `WebEditorBuilder`)

## v2.2 — Livré (2026-07) — durcissement production (#28)

### Early access (cœur local)
- [x] Synapse Studio + runtime unifié (cible Windows x64 + Vulkan)
- [x] API plugin C# (chargement DLL sandboxé, surface encore mince)
- [x] Mode headless benchmarks + reproductibilité (`SYNAPSE_SEED`)
- [x] Import FBX ASCII + USD USDA (limité)
- [x] Export scène glTF (entités + métadonnées Synapse)
- [x] Blueprint runtime (exécution graphe en simulation)
- [x] Marketplace lois (`.synapse-law`)
- [x] Captures PNG Studio (`--screenshot`, Avalonia Headless + Skia)
- [x] Matrice de maturité honnête (`docs/MATURITY.md`, `FeatureMaturityCatalog`)
- [x] Scène lab `samples/lab-heat-agents.synapse` exercée en tests
- [x] `FeatureMaturityCatalog` + `docs/MATURITY.md`
- [x] `Synapse.Web.Studio` (Blazor WASM) + `WasmStudioPublisher`
- [x] Collaboration runtime (`EngineHost.Collaboration`, `ScenePatchCodec`)

### Experimental (scaffolds — ne pas vendre comme production)
- [x] P2P multi-pairs TCP labo (`MultiPeerSimulationHub`)
- [x] P2P WAN scaffold (rendez-vous + STUN/TURN + AES-GCM)
- [x] OpenXR scaffold (détection loader + swapchain natif/simulé)
- [x] Éditeur web glTF / WebGPU preview (`site/editor/`)

## v2.11+ — Prochaine release

### Qualité
- [ ] Promouvoir au moins une surface cœur en tier **Supported** (Windows+GPU validé)
- [ ] Couverture de code > 50 %
- [ ] Réduction nette des warnings analyseurs sur Runtime / Physics / Core
- [ ] Plus de scènes `samples/` couvertes en CI (joints, lois, agents)

### Honnêteté produit
- [ ] Retirer ou remplacer les scaffolds Experimental avant toute claim « production »
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire actif

### Qualité
- [ ] Validation GPU manuelle matrice iGPU/discret (Windows/Linux/macOS)
- [ ] Couverture de code > 80 %
- [ ] Promouvoir au moins une surface cœur en tier **Supported** (Windows+GPU validé)

### Moteur
- [ ] Path tracing L-DNN teacher offline (batch)
- [ ] Native OpenUSD C++/Hydra plugin (optional) / full MDL SDK evaluation

## v2.8 — Moyen terme

### Rendu
- [ ] Path tracing L-DNN teacher offline (batch)
- [ ] DLSS / FSR upscaling neural
- [ ] Volumétriques avancées (fumée, incendie SPH couplé)

### Physique
- [ ] FEM tetrahedral (Corotational + Neo-Hookean)
- [ ] Fluides incompressibles (LBM GPU)
- [ ] Couplage thermo-mécanique bidirectionnel

### IA & simulation
- [ ] NEAT-G multi-objectif (forme + comportement)
- [ ] Behavior trees éditables dans Studio
- [ ] Jumeaux numériques persistants (export/import)

## v3.0 — Long terme (vraies intégrations)

- [ ] Éditeur web complet (WASM + WebGPU) au-delà du preview
- [ ] Réseau P2P collaboratif multi-pairs **hors labo** (remplace le scaffold WAN)
- [ ] Marketplace d'assets et de lois avec signatures

## Comment contribuer à la roadmap

1. Ouvrir une [Discussion](https://github.com/QuantumHacker10/Synapse/discussions) avec le tag **Roadmap**
2. Décrire le cas d'usage et la priorité perçue
3. Les items validés sont ajoutés ici et trackés comme issues

## Légende des priorités

| Symbole | Signification |
|---|---|
| ⭐⭐⭐⭐ | Critique — prochain sprint |
| ⭐⭐⭐ | Important — release courante |
| ⭐⭐ | Souhaitable — release suivante |
| ⭐ | Exploration — sans engagement de date |

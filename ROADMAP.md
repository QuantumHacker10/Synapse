# Roadmap — Synapse OMNIA

Vision publique du projet. L'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).

**Maturité :** v2.2 = accès anticipé / R&D avancée — voir [docs/MATURITY.md](docs/MATURITY.md).
Les items « Experimental » ne sont **pas** des capacités production.
## v2.4 — Actuel (2026-07) — Stack cinématique native

- [x] G-DNN full-res material resolve (`MeshletMaterialResolvePass`, `NaniteCinematicResolve`)
- [x] Mesh-shader compat compute (`MeshShaderCompatGenerator`)
- [x] L-DNN cinematic GI (`LumenCinematicGi` surface cache + path-trace blend)
- [x] Upscaling FrameGraph (`UpscalePass` : FSR / DLSS-compatible / MetalFX-compatible)
- [x] Continuum scène (`GpuContinuumScheduler` SPH + LBM + élasticité, scales Demo→Cinematic)
- [x] `EngineHost.EnableCinematicStack()` branché au present path

## v2.3 — Pipeline industriel

- [x] Cascade unifiée **LLM → Physics → Rendering → Simulation** (`OmniaIndustrialPipeline`)
- [x] Parsing world-delta LLM (éclairage, SDF, lois vivantes, matériaux, impulsions, BT)
- [x] Couplage Physics → L-DNN (`PhysicsFieldGiCoupler` + Lumen Neural 3.0)
- [x] Actuateur Simulation → Physics (`PhysicsActuator` heat/impulse)
- [x] **G-DNN Nanite Neural 3.0** (continuous LOD, densités meshlets, resolve cluster)
- [x] **L-DNN Lumen Neural 3.0** (surface radiance cache, multi-bounce, thermo-volumétrique)
- [x] FrameOrchestrator : Physics → Simulation → Coupling → Render → Quality
- [x] Continuum SPH / élasticité warm-ready (activation via `enableModules` LLM)
- [x] Studio Apply = `ApplyLlmWorldDelta`

## v2.0 — Fondations (2026-07)

## v2.2 — Actuel (2026-07)

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
- [x] Découpage `LivingLawCompiler.cs` (68 fichiers)
- [x] Couverture de code > 40 % (seuil CI Codecov)
- [x] Fondations P2P, VR (OpenXR), preview web
- [x] Tutoriels écrits + guide benchmarks (docs/)

## v2.1 — (2026-07)

### Experimental (scaffolds — ne pas vendre comme production)
- [x] P2P multi-pairs TCP labo (`MultiPeerSimulationHub`)
- [x] P2P WAN scaffold (rendez-vous **loopback** + AES-GCM)
- [x] OpenXR scaffold (détection loader + swapchain **synthétique**)
- [x] Éditeur web glTF / WebGPU preview (`site/editor/`)

## v2.3 — Durcissement (prochaine priorité)

### Qualité
- [ ] Promouvoir au moins une surface cœur en tier **Supported** (Windows+GPU validé)
- [ ] Couverture de code > 50 %
- [ ] Réduction nette des warnings analyseurs sur Runtime / Physics / Core
- [ ] Plus de scènes `samples/` couvertes en CI (joints, lois, agents)
## v2.2 — (2026-07)

- [x] Captures PNG live Studio (`--screenshot`, Avalonia Headless + Skia)
- [x] OpenXR swapchain Vulkan réel (`OpenXrVulkanSwapchain`)
- [x] P2P WAN (NAT rendezvous UDP + chiffrement AES-GCM)
- [x] Éditeur web glTF interactif (`site/editor/`, WebGPU)

## v2.4 — Communauté & qualité

### Honnêteté produit
- [ ] Retirer ou remplacer les scaffolds Experimental avant toute claim « production »
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire actif

### Moteur
### Qualité
- [ ] Couverture de code > 50 %
- [ ] Import USD binaire (USDC)

### Studio
- [ ] Blueprint éditable temps réel dans Studio
- [ ] Import USD binaire (USDC)
- [ ] Plugin marketplace hébergé

## v2.5 — Moyen terme

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

- [ ] Réseau P2P collaboratif multi-pairs **hors labo** (remplace le scaffold WAN)
- [ ] Support VR OpenXR **natif** (swapchain Vulkan réel, pas synthétique)
- [ ] Éditeur web complet (WASM + WebGPU) au-delà du preview
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

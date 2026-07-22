# Roadmap — Synapse OMNIA

Vision publique du projet. L'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).

## v2.3 — Actuel (2026-07) — Pipeline industriel

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

- [x] API plugin C# (chargement DLL sandboxé)
- [x] Mode headless benchmarks automatisés
- [x] Reproductibilité simulation (`SYNAPSE_SEED`)
- [x] Import FBX ASCII + USD USDA
- [x] Export scène glTF (entités + métadonnées Synapse)
- [x] Blueprint runtime (exécution graphe en simulation)
- [x] Marketplace lois (`.synapse-law`)
- [x] Découpage `LivingLawCompiler.cs` (68 fichiers)
- [x] Couverture de code > 40 % (seuil CI Codecov)
- [x] Fondations P2P, VR (OpenXR), preview web
- [x] Tutoriels écrits + guide benchmarks (docs/)

## v2.1 — (2026-07)

- [x] Captures PNG Studio dans README (`docs/screenshots/*.png`)
- [x] Découpage `LivingLawLibrary.cs` et `EntityBehaviorSystem.cs`
- [x] P2P multi-pairs (`MultiPeerSimulationHub`)
- [x] OpenXR session production (`OpenXrVulkanSession`)
- [x] Éditeur web WebGPU (`site/editor/`, `WebEditorBuilder`)

## v2.2 — (2026-07)

- [x] Captures PNG live Studio (`--screenshot`, Avalonia Headless + Skia)
- [x] OpenXR swapchain Vulkan réel (`OpenXrVulkanSwapchain`)
- [x] P2P WAN (NAT rendezvous UDP + chiffrement AES-GCM)
- [x] Éditeur web glTF interactif (`site/editor/`, WebGPU)

## v2.4 — Communauté & qualité

### Communauté
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire actif

### Qualité
- [ ] Couverture de code > 50 %
- [ ] Import USD binaire (USDC)

### Studio
- [ ] Blueprint éditable temps réel dans Studio
- [ ] Plugin marketplace hébergé

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

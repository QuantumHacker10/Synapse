# Roadmap — Synapse OMNIA

Vision publique du projet. Les dates sont indicatives ; l'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).

## v2.4 — Actuel (2026-07) — production-ready desktop

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

## v2.2 — Livré (2026-07)

- [x] Captures PNG live Studio (`--screenshot`, Avalonia Headless + Skia)
- [x] OpenXR swapchain Vulkan réel (`OpenXrVulkanSwapchain`)
- [x] P2P WAN (NAT rendezvous UDP + chiffrement AES-GCM)
- [x] Éditeur web glTF interactif (`site/editor/`, WebGPU)

## v2.4+ — Prochaine release

### Communauté
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire actif

### Qualité
- [ ] Validation GPU manuelle matrice iGPU/discret (Windows/Linux/macOS)
- [ ] Couverture de code > 80 %

### Moteur
- [ ] Plugin marketplace hébergé distant
- [ ] OpenXR compositor Vulkan réel (hors handles simulés)
- [ ] WAN NAT hors-loopback
- [ ] Path tracing L-DNN teacher offline (batch)

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

## v3.0 — Long terme

- [ ] Réseau P2P complet (simulations collaboratives multi-pairs)
- [ ] Support VR OpenXR production
- [ ] Éditeur web complet (WASM + WebGPU)
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

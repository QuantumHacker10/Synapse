# Roadmap — Synapse OMNIA

Vision publique du projet. Les dates sont indicatives ; l'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).

## v2.0 — Actuel (2026-07)

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

## v2.1 — Actuel (2026-07)

- [x] Captures PNG Studio dans README (`docs/screenshots/*.png`)
- [x] Découpage `LivingLawLibrary.cs` et `EntityBehaviorSystem.cs`
- [x] P2P multi-pairs (`MultiPeerSimulationHub`)
- [x] OpenXR session production (`OpenXrVulkanSession`)
- [x] Éditeur web WebGPU (`site/editor/`, `WebEditorBuilder`)

## v2.2 — Prochaine release

### Communauté
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire actif
- [ ] Captures PNG réelles Studio dans le README

### Qualité
- [ ] Découpage `LivingLawLibrary.cs` et `EntityBehaviorSystem.cs`
- [ ] Couverture de code > 50 %
- [ ] Import USD binaire (USDC)

### Moteur
- [ ] Blueprint éditable temps réel dans Studio
- [ ] Plugin marketplace hébergé
- [ ] Éditeur web WASM + WebGPU (preview interactif)

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

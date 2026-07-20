# Roadmap — Synapse OMNIA

Vision publique du projet. Les dates sont indicatives ; l'avancement réel est suivi via [GitHub Issues](https://github.com/QuantumHacker10/Synapse/issues) et [Discussions](https://github.com/QuantumHacker10/Synapse/discussions).

## v1.2 — Actuel (2026-07)

- [x] Synapse Studio (Avalonia + viewport Vulkan)
- [x] Physique industrielle (rigid bodies, joints, CCD, mesh collision)
- [x] G-DNN + L-DNN (SDF neuronaux, GI GPU-résidente)
- [x] Living laws (100 lois, hot-reload)
- [x] NEAT-G (évolution de formes)
- [x] HybridLlmRouter (ONNX, Ollama, cloud)
- [x] Multiplateforme (Windows, Linux, macOS)
- [x] CI : tests, Coverlet, CodeQL, release automatique

## v1.3 — Prochaine release

### Documentation & communauté
- [ ] Captures d'écran Studio dans le README
- [ ] Tutoriels vidéo (YouTube)
- [ ] Serveur Discord communautaire
- [ ] Contributing guide enrichi (architecture, revue de code)

### Qualité & maintenance
- [ ] Découpage progressif des monolithes (Physics, AI, Rendering)
- [ ] Couverture de code > 40 % (seuil CI)
- [ ] macOS dans la matrice de tests CI
- [ ] Dependabot + mises à jour Avalonia trimestrielles

### Moteur
- [ ] Import FBX/USD
- [ ] Blueprint runtime (exécution graphe en simulation)
- [ ] Export scène glTF complet (entités + matériaux L-DNN)
- [ ] Mode headless pour benchmarks automatisés

## v1.4 — Moyen terme

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

## v2.0 — Long terme

- [ ] API plugin C# (chargement DLL sandboxé)
- [ ] Réseau P2P (simulations collaboratives)
- [ ] Support VR (OpenXR + Vulkan)
- [ ] Éditeur web (WASM + WebGPU preview)
- [ ] Marketplace d'assets et de lois (.synapse-law)

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

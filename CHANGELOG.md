# Changelog

Toutes les modifications notables de **Synapse OMNIA** sont documentées ici.

Le format suit [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) et le projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Non publié]

### Ajouté

- **Synapse Studio** : viewport Vulkan embarqué (HWND), grille/gizmos, outils sélection/déplacement/rotation, édition d'entités et blueprint graphique.
- Lecture G-buffer GPU pour l'illumination L-DNN (fallback constantes si indisponible).
- **Physique industrielle** : `RigidBodyWorld` (primitives, AABB, contacts analytiques, PGS, sleep) et `MultiphysicsOrchestrator` (pas fixe, living laws + rigid bodies + continuum optionnel), branchés dans `EngineHost.TickPhysics`.
- **Rendu industriel** : SSAO réel (`LDNNBridge.ComputeAO`), kernels compute CPU (`ssao` / `blur_ao` / `downsample_irradiance`), simplification mesh QEM (`QuadricMeshSimplifier`).
- Tests de validation industrielle (repos, moment, SSAO, LOD, tick runtime).

### Modifié

- Interface Studio entièrement en français (menus, statuts, dialogues, console LLM).
- Texte « À propos » enrichi (onglet Performance) aligné sur le README et le site vitrine.
- README, site GitHub Pages et CHANGELOG recentrés sur le positionnement « moteur de simulation 3D ».
- Remplacement de la licence MIT par une **licence propriétaire** (anti-copie, anti-fork, anti-plagiat) — voir [LICENSE](LICENSE) et [COPYRIGHT](COPYRIGHT).
- Remplacement du LOD stochastique placeholder et des stubs AO / compute no-op.

### Supprimé

- Propriétés ViewModel mortes (`InspectorText`, `PlayPauseLabel`) et détails techniques superflus sur le site vitrine.

### À faire

- Développements en cours sur les branches `feat/*` (voir [CONTRIBUTING.md](CONTRIBUTING.md)).

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

[1.1.0]: https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0

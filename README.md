# SYNAPSE OMNIA — Moteur de simulation 3D · v1.1

[![Build](https://github.com/QuantumHacker10/Synapse/actions/workflows/build.yml/badge.svg)](https://github.com/QuantumHacker10/Synapse/actions/workflows/build.yml)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![.NET](https://img.shields.io/badge/.NET-10.0-512bd4)](global.json)

**Synapse OMNIA** est un moteur de **simulation 3D** : un monde numérique que l'on observe,
modifie et fait évoluer — pas une boîte à monter des niveaux de jeu.
**Synapse Studio** en est l'atelier pour éditer une scène, lancer la simulation et voir
comment formes, lois et habitants changent ensemble.

Là où les outils 3D classiques *figent* des objets et *rejouent* des règles immuables,
Synapse *apprend*, *réécrit* et *cultive* le monde simulé.

> **Produit v1.1** — Synapse Studio + un runtime unifié (physique, simulation, évolution,
> assistance créative, rendu temps réel). Build officiel **Windows x64** ; Linux et macOS
> en compilation locale.

**Site vitrine :** [quantumhacker10.github.io/Synapse](https://quantumhacker10.github.io/Synapse/) · **Releases :** [Télécharger v1.1](https://github.com/QuantumHacker10/Synapse/releases)

## Sommaire

- [Pourquoi Synapse ?](#pourquoi-synapse-)
- [Prérequis](#prérequis)
- [Démarrage rapide](#démarrage-rapide)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Pipeline G-DNN + L-DNN](#pipeline-g-dnn--l-dnn)
- [Synapse Studio](#synapse-studio)
- [Publish](#publish-windows-x64)
- [Tests & CI](#tests--ci)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Pourquoi Synapse ?

| Ailleurs (Unity, Unreal, Godot…) | Ici |
|---|---|
| Des scènes assemblées à la main | Un monde simulé qui évolue |
| Formes figées, découpées en triangles | Formes apprises, continues, zoomables sans limite |
| Physique gravée une fois pour toutes | Lois réécrivables pendant que la simulation tourne |
| Chaque objet modélisé individuellement | Populations de formes qui mutent et se sélectionnent |
| IA souvent imposée depuis le cloud | Assistance locale ou distante, selon vos contraintes |
| Entités scriptées | Habitants qui perçoivent, décident et s'adaptent |

Six idées rares réunies dans **un seul moteur de simulation**, pas comme des plugins séparés.

## Prérequis

| Composant | Version / détail |
|---|---|
| [.NET SDK](https://dotnet.microsoft.com/download) | **10.0.300** (voir [`global.json`](global.json)) |
| GPU | Pilote **Vulkan** à jour (NVIDIA, AMD, Intel ; MoltenVK sur macOS) |
| Windows (publish) | `glfw3.dll` 3.4+ (voir [glfw3.dll](#glfw3dll)) |
| LLM (optionnel) | [Ollama](https://ollama.com/) en local, ou clés API cloud (voir [Configuration](#configuration)) |

**Plateformes cibles :** Windows x64 (publish officiel), Linux et macOS via compilation locale.

## Démarrage rapide

```bash
# Cloner et entrer dans le dépôt
git clone https://github.com/QuantumHacker10/Synapse.git
cd Synapse

dotnet build
dotnet test

# Lancer Synapse Studio (interface Avalonia)
dotnet run --project src/Synapse.Studio

# Mode moteur GLFW seul, sans UI (--glfw est un alias)
dotnet run --project src/Synapse.Studio -- --engine

# Charger la scène d'exemple
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse
```

### glfw3.dll

Placez `glfw3.dll` (GLFW 3.4+) à côté de l'exécutable, ou dans
[`src/Synapse.Studio/native/`](src/Synapse.Studio/native/README.md) avant le publish.

## Configuration

| Source | Paramètres |
|---|---|
| [`src/Synapse.Studio/appsettings.json`](src/Synapse.Studio/appsettings.json) | Résolution, qualité, budgets physique/sim, LLM par défaut |
| CLI | `--width`, `--height`, `--scene`, `--quality`, `--validation` / `--no-validation`, `--engine` / `--glfw` |
| Variables d'environnement | `SYNAPSE_WIDTH`, `SYNAPSE_HEIGHT`, `SYNAPSE_SCENE` |
| LLM (jamais en dur dans le dépôt) | `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY`, `OLLAMA_HOST` |

Le routeur [`HybridLlmRouter`](src/Synapse.LLM/HybridLlmRouter.cs) bascule automatiquement entre ONNX, Ollama, OpenAI, Anthropic, Gemini et Azure selon disponibilité, coût et confidentialité.

## Architecture

Dix projets sous `src/`, tests sous `tests/` (solution [`Synapse.slnx`](Synapse.slnx)), scène d'exemple sous [`samples/`](samples/).

| Projet | Rôle |
|---|---|
| `Synapse.Core` | Fondations mathématiques / physiques (`PhysicsState` 256 octets, algèbre, octree, kd-tree, sécurité) |
| `Synapse.Physics` | `LivingLawCompiler` — 100 lois texte sur 18 domaines, hot-reload ; solveurs Maxwell, SPH, Lattice-Boltzmann, Schrödinger, N-corps, champs stochastiques |
| `Synapse.AI` | `NeatGEvolutionEngine` — évolution NEAT-G, sélection NSGA-II, fitness SDF + irradiance L-DNN |
| `Synapse.Genomics` | `GeoGenome` — génomes de formes (builder, validation, registry, pool) |
| `Synapse.Rendering` | Vulkan RHI, G-DNN (SDF), L-DNN (GI neuronale), polygonisation LOD, mesh→SDF, export glTF, styles artistiques |
| `Synapse.LLM` | `HybridLlmRouter` — ONNX / Ollama / OpenAI / Anthropic / Gemini / Azure + parse lighting/SDF |
| `Synapse.Simulation` | `SentienceManager` — entités, behavior trees, perception, jumeaux numériques |
| `Synapse.Infrastructure` | Qualité adaptative, benchmarks, logging et config |
| `Synapse.Runtime` | `EngineHost` + `FrameOrchestrator` + projets `.synapse`, application des hints LLM→scène |
| `Synapse.Studio` | **Synapse Studio** — éditeur Avalonia + mode `--engine` GLFW |

```mermaid
flowchart LR
  Studio[Synapse Studio] --> Runtime[EngineHost]
  Runtime --> Physics[Living Laws]
  Runtime --> AI[NEAT-G]
  Runtime --> Render[G-DNN + L-DNN Vulkan]
  Runtime --> Sim[Sentience]
  Studio --> LLM[HybridLlmRouter]
  LLM --> Runtime
```

## Pipeline G-DNN + L-DNN

| Domaine | Capacités |
|---|---|
| **G-DNN (géométrie)** | SDF neuronaux, broad-phase BVH (`AABBTree`) pour le ray marching, polygonisation LOD adaptative, cache disque des chaînes polygonisées, pipeline mesh→SDF (`MeshToSdfPipeline`), export glTF/GLB |
| **L-DNN (éclairage)** | GI hybride (SSGI + cascades + MLP), teacher path tracing online, ombres neuronales, reflets/réfractions neuronaux, brouillard froxel + nuages procéduraux, profils Tiny/Small/Full, cache GI scènes statiques |
| **Intégration** | G-Buffer étendu (velocity + material ID), shadow pass dans la frame, RT hybride branché, styles post (Cartoon / Grayscale / Noir) |
| **Studio / LLM** | Console LLM → parse JSON lighting/SDF → lumières L-DNN, fog/nuages, entités scène (`ApplyLlmSceneHints`) |

## Synapse Studio

Atelier pour explorer et piloter la simulation :

- **Vue 3D temps réel** — viewport Vulkan embarqué (Windows HWND) avec grille, gizmos et outils d'édition (sélection, déplacement, rotation)
- **Projets `.synapse`** — ouvrir, sauver et organiser vos scènes
- **Lois physiques** — réécrire les règles du monde sans arrêter la simulation
- **Évolution** — faire muter des populations de formes, lancer des agents, play/pause (Espace)
- **Console créative** — décrire une scène en langage naturel et voir le monde réagir
- **Outils d'édition** — blueprints graphiques, sculpt, import d'assets Megascans
- **Tableau de bord** — cadence (IPS), charge physique/sim, qualité adaptative et GI L-DNN

## Publish (Windows x64)

```bash
dotnet publish src/Synapse.Studio/Synapse.Studio.csproj -c Release -r win-x64 --self-contained true -o artifacts/Synapse-win-x64
```

Les tags `v*` déclenchent [`.github/workflows/release.yml`](.github/workflows/release.yml) et publient un zip win-x64 sur GitHub Releases.

## Tests & CI

```bash
dotnet test
```

Suite xUnit + FluentAssertions sous [`tests/Synapse.Tests`](tests/Synapse.Tests) : Core, Physics, AI, Genomics, Rendering/G-DNN, L-DNN, LLM, Simulation, Runtime.

| Workflow | Rôle |
|---|---|
| [`build.yml`](.github/workflows/build.yml) | Ubuntu — tests + Coverlet ; Windows — publish artefact |
| [`analysis.yml`](.github/workflows/analysis.yml) | Analyseurs + `dotnet format --verify-no-changes` |
| [`release.yml`](.github/workflows/release.yml) | Zip win-x64 sur tag `v*` |
| [`pages.yml`](.github/workflows/pages.yml) | Déploiement du site vitrine sur GitHub Pages |

## Contribuer

Voir **[CONTRIBUTING.md](CONTRIBUTING.md)** pour le flux Git complet :

- Branches `feat/*` → `develop` → `main` (PR obligatoires sur `main`)
- [CHANGELOG.md](CHANGELOG.md) pour l'historique des versions
- Tags `v*` (ex. `v1.1.0`) pour les releases — voir [releases](https://github.com/QuantumHacker10/Synapse/releases)

En bref :

1. Forker, créer une branche depuis `develop` (`feat/ma-fonctionnalite`)
2. `dotnet build && dotnet test` — la CI doit passer
3. Mettre à jour le CHANGELOG si le changement est visible
4. Ouvrir une pull request vers `develop`

Les issues et discussions GitHub sont ouvertes pour bugs, idées et questions d'architecture.

## Site

Page vitrine dans [`site/`](site/) — présentation du moteur de simulation 3D, en langage clair,
avec lien de téléchargement Releases.
Déployée automatiquement sur [GitHub Pages](https://quantumhacker10.github.io/Synapse/) à chaque push touchant `site/**`.

## Licence

**Licence propriétaire — tous droits réservés.** Ce dépôt n'est pas open source.

Interdit sans autorisation écrite : copie, fork, redistribution, plagiat, œuvres
dérivées et usage commercial. Consultation du source et exécution des binaires
[Releases](https://github.com/QuantumHacker10/Synapse/releases) officiels
autorisées uniquement à titre personnel et non commercial.

Voir [`LICENSE`](LICENSE) et [`COPYRIGHT`](COPYRIGHT).

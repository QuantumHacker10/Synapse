# SYNAPSE OMNIA — GDNN Synapse Engine 1.1

Moteur de simulation et éditeur **G-DNN Studio** en **C# 14 / .NET 10**.
Les moteurs 3D classiques *assemblent et simulent* ; Synapse *apprend, réécrit et cultive*
le monde simulé — formes neuronales (SDF), lois physiques vivantes, géométrie évolutive
(NEAT-G), agents sentients, routeur LLM multi-fournisseurs, rendu **Vulkan**.

> **Produit v1.1** — Studio Avalonia + runtime unifié (physique, simulation, AI, LLM, rendu).
> Surfaces Vulkan : **Windows** (HWND + GLFW), **Linux** / **macOS** (GLFW + MoltenVK).

## Pourquoi Synapse ?

| Ailleurs (Unity, Unreal, Godot…) | Ici |
|---|---|
| Forme = maillage figé de triangles | Forme = fonction apprise (SDF neuronal), continue et sans résolution fixe |
| Physique câblée une fois pour toutes | 100 lois en texte sur 18 domaines, compilées, versionnées, rechargées à chaud |
| L’artiste modélise chaque objet à la main | Populations de formes évoluées (NEAT-G) puis sélectionnées |
| IA générative souvent = un service cloud unique | Aiguilleur LLM local (Ollama / ONNX) + cloud, selon coût et confidentialité |
| Rendu et IA dans des outils séparés | Vulkan différé + ray tracing BLAS/TLAS + illumination neuronale L-DNN dans le même runtime |
| PNJ scriptés | Agents sentients : perception, behavior trees, ordonnanceur actif/dormant sous budget frame |

Six idées rares réunies dans **un seul runtime** .NET, pas comme des outils séparés.

## Démarrage rapide

```bash
# Prérequis : SDK .NET 10.0.300 (global.json), GPU Vulkan, glfw3.dll (voir ci-dessous)

dotnet build
dotnet test

# Lancer G-DNN Studio (interface Avalonia)
dotnet run --project src/Synapse.Studio

# Mode moteur GLFW seul, sans UI (--glfw est un alias)
dotnet run --project src/Synapse.Studio -- --engine

# Charger la scène d'exemple
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse
```

### glfw3.dll

Placez `glfw3.dll` (GLFW 3.4+) à côté de l’exécutable, ou dans
[`src/Synapse.Studio/native/`](src/Synapse.Studio/native/README.md) avant le publish.

### Configuration

- [`src/Synapse.Studio/appsettings.json`](src/Synapse.Studio/appsettings.json) — résolution, qualité, budgets
- CLI : `--width`, `--height`, `--scene`, `--quality`, `--validation` / `--no-validation`, `--engine` / `--glfw`
- Env : `SYNAPSE_WIDTH`, `SYNAPSE_HEIGHT`, `SYNAPSE_SCENE`
- LLM : variables `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY`, `OLLAMA_HOST` (jamais en dur dans le dépôt)

## Architecture

Dix projets sous `src/`, tests sous `tests/` (solution `Synapse.slnx`), scène d'exemple sous [`samples/`](samples/).

| Projet | Rôle |
|---|---|
| `Synapse.Core` | Fondations mathématiques / physiques (`PhysicsState` 256 octets, algèbre, octree, kd-tree, sécurité) |
| `Synapse.Physics` | `LivingLawCompiler` — 100 lois texte sur 18 domaines, hot-reload ; solveurs Maxwell, SPH, Lattice-Boltzmann, Schrödinger, N-corps, champs stochastiques |
| `Synapse.AI` | `NeatGEvolutionEngine` — évolution NEAT-G, sélection NSGA-II, kernels neuronaux |
| `Synapse.Genomics` | `GeoGenome` — génomes de formes (builder, validation, registry, pool) |
| `Synapse.Rendering` | Vulkan RHI, G-DNN (SDF neuronaux), ray tracing BLAS/TLAS, L-DNN, polygonisation neuronale, matériaux Substrate, streaming assets |
| `Synapse.LLM` | `HybridLlmRouter` — ONNX / Ollama / OpenAI / Anthropic / Gemini / Azure |
| `Synapse.Simulation` | `SentienceManager` — entités, behavior trees, perception, jumeaux numériques |
| `Synapse.Infrastructure` | Qualité adaptative, benchmarks, **logging** et **config** |
| `Synapse.Runtime` | `EngineHost` + `FrameOrchestrator` + projets `.synapse`, blueprints, sculpt |
| `Synapse.Studio` | **G-DNN Studio** Avalonia + mode `--engine` GLFW |

## Fonctionnalités Studio

- Viewport Vulkan (HWND embarqué ou fenêtre GLFW)
- Scene explorer / inspector, projets `.synapse` (New / Open / Save)
- Éditeur de living laws avec hot-reload
- Évolution NEAT-G, spawn d’agents, play/pause simulation
- Console LLM (Ollama local ; le routeur `HybridLlmRouter` gère aussi les fournisseurs cloud si clés présentes)
- Blueprint editor, sculpt strokes, import Megascans
- HUD performances (FPS, budgets physique / sim, qualité adaptative)

## Publish (Windows x64)

```bash
dotnet publish src/Synapse.Studio/Synapse.Studio.csproj -c Release -r win-x64 --self-contained true -o artifacts/Synapse-win-x64
```

Les tags `v*` déclenchent [`.github/workflows/release.yml`](.github/workflows/release.yml).

## Tests & CI

```bash
dotnet test
```

Suite xUnit + FluentAssertions (~130 tests) sous [`tests/Synapse.Tests`](tests/Synapse.Tests) : Core, Physics, AI, Genomics, Rendering/G-DNN, LLM, Simulation, Runtime.

- `build.yml` — Ubuntu tests + Coverlet ; Windows publish artefact
- `analysis.yml` — analyseurs + `dotnet format --verify-no-changes`
- `release.yml` — zip win-x64 sur tag
- `pages.yml` — déploiement du site sur GitHub Pages

## Site

Page vitrine dans [`site/`](site/) — présentation produit, différenciation et lien de téléchargement Releases.
Déployée automatiquement sur GitHub Pages ([`pages.yml`](.github/workflows/pages.yml)) à chaque push touchant `site/**`.

## Licence

MIT — voir [`LICENSE`](LICENSE).

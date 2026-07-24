# Benchmarks — Synapse OMNIA v2

Synapse v2 inclut un mode **headless** reproductible pour mesurer les performances simulation/physique sans GPU interactif.

## Lancer un benchmark

```bash
dotnet run --project src/Synapse.Studio -- --benchmark samples/benchmarks/default.json --benchmark-out results.json --seed 42 --headless
```

Ou via le moteur seul :

```bash
dotnet run --project src/Synapse.Studio -- --engine --headless --benchmark samples/benchmarks/default.json --seed 42
```

## Configuration (`samples/benchmarks/default.json`)

| Champ | Description |
|---|---|
| `warmupFrames` | Frames ignorées (JIT / stabilisation) |
| `measureFrames` | Frames mesurées |
| `simulationSeed` | Graine reproductible (`SYNAPSE_SEED`) |
| `scenePath` | Scène `.synapse` optionnelle |
| `activeLawId` | Loi physique active |

## Rapport JSON

Le fichier de sortie contient :

- `physicsMsAvg`, `simulationMsAvg`, `physicsMsP95`
- `entityCount`, `activeLawId`, `synapseVersion`
- `completedAt` (UTC)

## CI

Le workflow GitHub Actions exécute `dotnet test` avec couverture Codecov (**seuil 70 %**). Les benchmarks headless peuvent être ajoutés au pipeline via `--benchmark` sur les runners Linux.

## Reproductibilité

```bash
export SYNAPSE_SEED=42
dotnet test
```

Les runs avec la même graine produisent des séquences aléatoires identiques dans les modules utilisant `SimulationReproducibility`.

## Benchmark compilation des lois

Mesure combien de lois du catalogue runtime compilent **directement** vs via **fallback de catégorie** :

```bash
dotnet run --project src/Synapse.Studio -- --law-benchmark --law-benchmark-out law-compilation-report.json
```

### Rapport JSON (`LawCompilationBenchmarkReport`)

| Champ | Description |
|---|---|
| `totalLaws` | Nombre de lois dans `LawLibrary.LoadBuiltIn()` |
| `compiledDirect` | Compilation verbatim réussie |
| `compiledWithFallback` | Fallback par catégorie (ex. `alpha * laplacian(T)`) |
| `failed` | Échecs de compilation |
| `directCompileRate` | Ratio direct / total |
| `entries[]` | Détail par loi (`lawId`, `category`, `usedFallback`) |

## Régression visuelle GI (golden hash)

Test CI `GiGoldenSnapshotTests` compare un hash SHA-256 deterministe du champ d'irradiance L-DNN (preview procédurale 32×32).

Mettre à jour la référence après un changement intentionnel du pipeline GI :

```bash
SYNAPSE_UPDATE_GI_GOLDEN=1 dotnet test tests/Synapse.Tests --filter GiGoldenSnapshot
```

Fichier de référence : `tests/Synapse.Tests/Rendering/Golden/gi-procedural-32x32.sha256`

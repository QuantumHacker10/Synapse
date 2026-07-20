# Tutoriels — Synapse OMNIA v2

Guides pas-à-pas pour démarrer sans friction.

## 1. Premier lancement (5 min)

```bash
git clone https://github.com/QuantumHacker10/Synapse.git
cd Synapse
dotnet build
dotnet test
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse
```

**Prérequis :** [.NET SDK 10.0.300](https://dotnet.microsoft.com/download), GPU Vulkan.

## 2. Mode headless (étudiants / CI)

Sans fenêtre graphique :

```bash
dotnet run --project src/Synapse.Studio -- --engine --headless --seed 42
```

120 ticks simulation sont exécutés puis le processus se termine.

## 3. Benchmark reproductible

Voir [BENCHMARKS.md](BENCHMARKS.md).

## 4. Exporter une scène en glTF

```bash
dotnet run --project src/Synapse.Studio -- --engine --headless \
  --scene samples/demo.synapse \
  --export-scene export/demo.gltf
```

## 5. Importer un mesh FBX / USD

Dans Studio ou via code :

```csharp
var loader = new GDNN.Rendering.MeshIO.MeshLoader();
var result = await loader.LoadAsync("samples/meshes/tetra.usda");
```

Formats supportés v2 : `.gltf`, `.glb`, `.obj`, `.fbx` (ASCII), `.usda`.

## 6. Installer une loi marketplace

```csharp
var market = new Synapse.Runtime.LawMarketplace();
var law = await market.ImportAsync("samples/laws/custom_heat_wave.synapse-law");
host.ApplyLaw(law.Id); // après enregistrement dans la bibliothèque
```

## 7. Plugin C#

Implémentez `Synapse.Plugins.ISynapsePlugin`, compilez en DLL, puis :

```bash
dotnet run --project src/Synapse.Studio -- --engine --plugin-dir samples/plugins
```

## 8. Blueprint runtime

```csharp
var executor = new BlueprintRuntimeExecutor(host, logger);
executor.Load(BlueprintDocument.CreateDefault());
await executor.TickAsync(0.016f);
```

## Ressources

- [GETTING_STARTED.md](../GETTING_STARTED.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [API.md](API.md)
- [Discussions GitHub](https://github.com/QuantumHacker10/Synapse/discussions)

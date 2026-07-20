# Getting started with Synapse

This tutorial walks you through building, running, and exploring your first Synapse
simulation. It assumes .NET 10 and a Vulkan-capable GPU (see the [README](../README.md)).

## 1. Clone and verify the toolchain

```bash
git clone https://github.com/QuantumHacker10/Synapse.git
cd Synapse

dotnet --version   # should match global.json (10.0.300)
dotnet build
dotnet test
```

If tests pass, the core modules (physics, G-DNN, L-DNN, AI, runtime) are healthy on your machine.

## 2. Launch Synapse Studio

Synapse Studio is the Avalonia-based workshop for editing scenes and driving the simulation.

```bash
dotnet run --project src/Synapse.Studio
```

You should see:

- A **3D viewport** (Vulkan) with grid and gizmos
- **Play / pause** (Space) to control the simulation clock
- Panels for entities, physical laws, evolution, and the LLM creative console

### Engine-only mode

To run the GLFW/Vulkan engine without the UI (useful for headless debugging or CI-like runs):

```bash
dotnet run --project src/Synapse.Studio -- --engine
```

## 3. Load the demo scene

The repository includes a sample project at [`samples/demo.synapse`](../samples/demo.synapse):

| Entity | Type | Role |
|---|---|---|
| Ground | Mesh | Static floor |
| Agent_Alpha / Agent_Beta | Character | Sentient agents (patrol / guard profiles) |
| NeuralForm | Genome | Evolvable neural SDF shape |

Load it from the CLI:

```bash
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse
```

Or set the environment variable:

```bash
export SYNAPSE_SCENE=samples/demo.synapse
dotnet run --project src/Synapse.Studio
```

The scene activates the **heat equation** living law by default (`activeLawId`).

## 4. Explore the simulation loop

Each frame, `EngineHost` orchestrates:

1. **Physics** — `MultiphysicsOrchestrator` advances rigid bodies, joints, and living laws
2. **Simulation** — `SentienceManager` updates agents, perception, and behavior trees
3. **Evolution** — `NeatGEvolutionEngine` may mutate shape genomes (NEAT-G + NSGA-II)
4. **Rendering** — G-DNN ray-marches neural SDFs; L-DNN computes GI, shadows, and SSAO

Press **Space** to pause and inspect the world without advancing time.

## 5. Rewrite a physical law (living laws)

Synapse can **hot-reload** textual physical laws while the simulation runs.

In Studio, open the **Physical laws** panel and select a different law from the
`LivingLawLibrary` (Maxwell, SPH, heat equation, etc.). The `LivingLawCompiler`
recompiles the law and applies it on the next physics tick — no restart required.

This is one of Synapse's core differentiators: rules are not frozen at build time.

## 6. Evolve a shape (G-DNN + NEAT-G)

The demo scene includes a **NeuralForm** entity backed by a `GeoGenome`. To experiment:

1. Open the **Evolution** panel in Studio
2. Start evolution — NEAT-G mutates SDF network weights and topology
3. Fitness combines SDF quality and L-DNN irradiance feedback
4. Watch the neural shape change over generations

Under the hood:

- `Synapse.Genomics` manages genome serialization and validation
- `Synapse.AI.NeatGEvolutionEngine` runs NSGA-II selection
- `Synapse.Rendering` G-DNN evaluates and polygonizes the SDF at adaptive LOD

## 7. Describe a scene with the LLM console (optional)

If you have a local [Ollama](https://ollama.com/) instance or cloud API keys configured,
open the **Creative console** and describe lighting or geometry in natural language.

The `HybridLlmRouter` parses structured JSON hints (lights, fog, SDF parameters) and
applies them via `ApplyLlmSceneHints` — bridging language models to L-DNN and G-DNN.

Configure keys via environment variables (never commit secrets):

```bash
export OLLAMA_HOST=http://localhost:11434
# or OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, …
```

## 8. Create your own `.synapse` project

A project file is JSON describing entities, camera, and active law:

```json
{
  "name": "My First Simulation",
  "version": "1.0",
  "activeLawId": "heat_equation",
  "entities": [
    {
      "id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      "name": "Floor",
      "type": "Mesh",
      "position": { "x": 0, "y": 0, "z": 0 },
      "scale": { "x": 5, "y": 0.1, "z": 5 },
      "visible": true
    }
  ],
  "camera": {
    "position": { "x": 0, "y": 2, "z": 4 },
    "yaw": -90,
    "pitch": -10,
    "fov": 60
  },
  "assets": {}
}
```

Save it as `my-scene.synapse` and load:

```bash
dotnet run --project src/Synapse.Studio -- --scene my-scene.synapse
```

## 9. Publish a standalone build

For a self-contained Windows x64 build:

```bash
dotnet publish src/Synapse.Studio/Synapse.Studio.csproj \
  -c Release -r win-x64 --self-contained true \
  -o artifacts/Synapse-win-x64
```

Multi-platform publish:

```bash
bash scripts/publish-all.sh
```

Official releases are tagged `v*` (e.g. `v1.1.0`, `v1.2.0`) and built automatically by CI.

## Next steps

- Read [architecture.md](architecture.md) for module diagrams and data flow
- Browse [`src/Synapse.Runtime/EngineHost.cs`](../src/Synapse.Runtime/EngineHost.cs) for the runtime facade
- Open a [GitHub issue](https://github.com/QuantumHacker10/Synapse/issues) for bugs or ideas
- Share demos (videos, GIFs) on [r/CSharp](https://reddit.com/r/csharp), Hacker News, or 3D engine forums

## Troubleshooting

| Symptom | Fix |
|---|---|
| Vulkan device lost / black viewport | Update GPU drivers; on macOS ensure MoltenVK is available |
| Missing `glfw3.dll` on Windows | Copy GLFW 3.4+ next to the exe ([native README](../src/Synapse.Studio/native/README.md)) |
| LLM console idle | Set `OLLAMA_HOST` or a cloud API key; check `appsettings.json` defaults |
| Tests fail on CI but pass locally | Run `dotnet format --verify-no-changes` and re-test |

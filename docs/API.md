# Référence des APIs publiques — Synapse OMNIA

Ce document décrit les points d'entrée principaux pour intégrer ou étendre Synapse. Les types internes et les monolithes de recherche ne sont pas listés ici.

## Synapse.Runtime — Orchestration

### `EngineHost`

Facade centrale du runtime. Point d'entrée pour Studio et intégrations embarquées.

| Membre | Description |
|---|---|
| `InitializeModules()` | Charge physique, LLM, évolution NEAT-G, sentience |
| `InitializeRender(w, h)` | Rendu GLFW/Vulkan |
| `InitializeRenderFromHwnd(hwnd, w, h)` | Viewport embarqué Windows |
| `TickPhysics(dt)` | Pas de simulation physique |
| `TickSimulationAsync()` | Tick sentience / behavior trees |
| `TickRender()` | Frame de rendu G-DNN + L-DNN |
| `CompileLaw(lawId, expression)` | Compile et active une loi physique |
| `ListLaws()` | Liste les 100 lois du catalogue |
| `LoadScene(path)` / `SaveScene(path)` | I/O projets `.synapse` |
| `Scene` | Document scène courant (`SceneDocument`) |
| `LawCompiler` | Compilateur de lois vivantes |
| `Multiphysics` | Orchestrateur multiphysique |
| `LlmRouter` | Routeur LLM multi-provider |
| `Sentience` | Gestionnaire d'entités sentientes |

```csharp
var host = new EngineHost(config, logger);
host.InitializeModules();
host.InitializeRender(1280, 720);
host.CompileLaw("heat_equation", "dT/dt = alpha * laplacian(T)");
```

### `SceneDocument`

Modèle de scène sérialisable (JSON `.synapse`) : entités, caméra, loi active, joints, assets.

### `FrameOrchestrator`

Coordonne l'ordre des ticks (physique → simulation → rendu) pour une frame complète.

---

## Synapse.Physics — Simulation

### `LivingLawCompiler`

Compile, valide et hot-reload des lois physiques textuelles.

| Méthode | Description |
|---|---|
| `LoadLibrary()` | Charge les 100 lois du catalogue |
| `Compile(expression)` | Compile une expression en bytecode |
| `Apply(lawId, field)` | Applique une loi sur le champ physique |
| `Validate(lawId)` | Vérifie cohérence dimensionnelle et stabilité |

Lois organisées par `LawCategory` : thermodynamique, mécanique des fluides, électromagnétisme, quantique, etc.

### `MultiphysicsOrchestrator`

Orchestre rigid bodies, living laws et continuum (Maxwell, SPH, LBM).

| Membre | Description |
|---|---|
| `RigidWorld` | Monde de corps rigides (PGS, CCD, sleep) |
| `Tick(dt)` | Pas de simulation complet |
| `AddEntity(desc)` | Ajoute une entité physique |

### `RigidBodyWorld`

Simulation de corps rigides avec joints et véhicules.

| Type | Description |
|---|---|
| `BodyType` | Static, Dynamic, Kinematic |
| `ColliderShape` | Sphere, Box, Capsule, Mesh, ConvexHull |
| `JointType` | Hinge, BallSocket, Slider, Fixed, Distance |
| `VehicleController` | Suspension raycast, steer/drive/brake |

### `PhysicsCertification`

Suite de validation industrielle (repos, moment, CCD, CFL, élasticité, SPH).

```csharp
var report = PhysicsCertification.RunIndustrialCore();
Console.WriteLine(report.Passed ? "OK" : report.Failures);
```

---

## Synapse.Core — Fondations

### `PhysicsState`

Struct de 256 octets représentant l'état thermodynamique et cinématique d'un point de champ.

| Champ | Type | Description |
|---|---|---|
| `Density`, `Pressure`, `Temperature` | `double` | Variables thermodynamiques |
| `Velocity`, `Position`, `HeatFlux` | `Vector3D` | Cinématique et transfert |
| `VelocityGradient` | `Tensor3D` | Gradient de vitesse |

Méthodes notables : équations d'état (idéale, Van der Waals, Peng-Robinson), propriétés de transport, intégration temporelle (RK4, Verlet).

### `Vector3D`, `QuaternionD`, `Tensor3D`

Algèbre 3D double précision utilisée dans tout le moteur.

### `Octree`, `KdTree`

Structures spatiales pour broad-phase et requêtes de voisinage.

---

## Synapse.Rendering — Rendu Vulkan

### `RenderEngine`

Moteur de rendu Vulkan principal.

| Capacité | Description |
|---|---|
| G-DNN | SDF neuronaux, ray marching, polygonisation LOD |
| L-DNN | GI hybride, ombres neuronales, SSAO, brouillard |
| Mesh pipeline | Import mesh → SDF, export glTF/GLB |
| Shader compilation | DXC/glslang → SPIR-V |

### `GpuResidentGiPipeline`

Pipeline GI GPU-résident (SSGI + cascades) sans fill constant.

---

## Synapse.AI — Évolution

### `NeatGEvolutionEngine`

Évolution NEAT-G de formes SDF avec sélection NSGA-II.

| Membre | Description |
|---|---|
| `RunGeneration()` | Lance une génération d'évolution |
| `BestFitness` | Meilleur score de fitness |
| `Population` | Population courante de génomes |

Fitness basée sur SDF + irradiance L-DNN.

---

## Synapse.Genomics — Formes

### `GeoGenome`

Génome de forme : builder, validation, registry, pool de mutations.

---

## Synapse.LLM — Intelligence

### `HybridLlmRouter`

Routeur multi-provider avec circuit breaker, rate limiting et fallback.

| Provider | Configuration |
|---|---|
| ONNX | Modèles locaux |
| Ollama | `OLLAMA_HOST` |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Gemini | `GEMINI_API_KEY` |
| Azure | `AZURE_OPENAI_API_KEY` |

| Méthode | Description |
|---|---|
| `RouteAsync(prompt, context)` | Routage et complétion texte |
| `RouteChatAsync(messages, context)` | Chat multi-tours |
| `InferStructuredAsync<T>(prompt, context)` | Sortie JSON typée |

---

## Synapse.Simulation — Entités

### `SentienceManager`

Gestion des entités sentientes : perception, behavior trees, jumeaux numériques.

| Membre | Description |
|---|---|
| `EntityCount` | Nombre d'entités actives |
| `TickAsync(dt)` | Tick simulation comportementale |
| `RegisterEntity(id, profile)` | Enregistre une entité |

---

## Synapse.Infrastructure — Config & qualité

### `SynapseConfig`

Configuration application : résolution, qualité, budgets physique/sim, LLM.

```csharp
var config = SynapseConfig.Load();
// ou depuis un fichier appsettings.json spécifique
var config = SynapseConfig.Load("appsettings.json");
```

### `RuntimeQualityManager`

Qualité adaptative : ajuste preset L-DNN selon charge GPU/CPU.

---

## Format de projet `.synapse`

```json
{
  "name": "string",
  "version": "string",
  "activeLawId": "string",
  "entities": [{ "id", "name", "type", "position", "scale", "visible", ... }],
  "camera": { "position", "yaw", "pitch", "fov" },
  "joints": [{ "type", "bodyA", "bodyB", "anchor", ... }],
  "assets": {}
}
```

Types d'entités : `Mesh`, `Character`, `Genome`, `Light`, `Vehicle`.

---

## Namespaces

| Namespace | Assembly | Rôle |
|---|---|---|
| `Synapse.Runtime` | Synapse.Runtime | Orchestration, scènes |
| `Synapse.Physics` | Synapse.Physics | Simulation, lois vivantes |
| `Synapse.Core` | Synapse.Core | Math, PhysicsState, structures spatiales |
| `GDNN.Rendering.Engine` | Synapse.Rendering | Rendu Vulkan |
| `GDNN.Llm` | Synapse.LLM | Routeur LLM |
| `GDNN.Scene` | Synapse.Runtime | Modèle de scène |
| `GDNN.Sentience` | Synapse.Simulation | Entités sentientes |
| `GDNN.Core.NEAT` | Synapse.AI | Évolution NEAT-G |
| `Synapse.Infrastructure.*` | Synapse.Infrastructure | Config, logging, qualité |

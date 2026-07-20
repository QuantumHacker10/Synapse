# Architecture — Synapse OMNIA

Diagrammes et vue d'ensemble de l'architecture du moteur de simulation 3D.

## Vue d'ensemble

```mermaid
flowchart TB
    subgraph Studio["Synapse Studio (Avalonia)"]
        UI[Interface utilisateur]
        VP[Viewport Vulkan embarqué]
        LLMConsole[Console LLM]
    end

    subgraph Runtime["Synapse.Runtime"]
        EH[EngineHost]
        FO[FrameOrchestrator]
        SD[SceneDocument]
    end

    subgraph Simulation["Couches de simulation"]
        PHY[MultiphysicsOrchestrator]
        AI[NeatGEvolutionEngine]
        SIM[SentienceManager]
    end

    subgraph Rendering["Synapse.Rendering"]
        GDNN[G-DNN — SDF neuronaux]
        LDNN[L-DNN — GI neuronale]
        VK[Vulkan RHI]
    end

    subgraph External["Services externes"]
        LLM[HybridLlmRouter]
        OLL[Ollama / OpenAI / Anthropic]
    end

    UI --> EH
    VP --> EH
    LLMConsole --> LLM
    LLM --> OLL
    LLM --> EH

    EH --> FO
    FO --> PHY
    FO --> AI
    FO --> SIM
    FO --> GDNN
    FO --> LDNN

    PHY --> LL[LivingLawCompiler]
    GDNN --> VK
    LDNN --> VK
    EH --> SD
```

## Pipeline par frame

```mermaid
sequenceDiagram
    participant S as Studio / Boucle
    participant E as EngineHost
    participant P as Multiphysics
    participant Sim as Sentience
    participant R as RenderEngine

    S->>E: TickPhysics(dt)
    E->>P: RigidWorld.Step + LivingLaw.Apply
    P-->>E: État champ mis à jour

    S->>E: TickSimulationAsync()
    E->>Sim: Behavior trees + perception
    Sim-->>E: Entités mises à jour

    S->>E: TickRender()
    E->>R: G-Buffer + G-DNN ray march
    R->>R: L-DNN GI + SSAO + post-process
    R-->>S: Frame présentée
```

## Modules et dépendances

```mermaid
flowchart LR
    Studio[Synapse.Studio] --> Runtime
    Runtime[Synapse.Runtime] --> Physics
    Runtime --> Rendering
    Runtime --> AI
    Runtime --> Simulation
    Runtime --> LLM
    Runtime --> Infrastructure

    Physics[Synapse.Physics] --> Core
    Rendering[Synapse.Rendering] --> Core
    AI[Synapse.AI] --> Core
    AI --> Genomics
    Simulation[Synapse.Simulation] --> Core
    LLM[Synapse.LLM] --> Core
    Genomics[Synapse.Genomics] --> Core
    Infrastructure[Synapse.Infrastructure] --> Core

    Core[Synapse.Core]
```

| Projet | Responsabilité | Dépend de |
|---|---|---|
| `Synapse.Core` | Math, PhysicsState, octree, kd-tree | — |
| `Synapse.Physics` | Living laws, rigid bodies, multiphysique | Core |
| `Synapse.Rendering` | Vulkan, G-DNN, L-DNN, shaders | Core |
| `Synapse.AI` | NEAT-G, évolution | Core, Genomics |
| `Synapse.Genomics` | Génomes de formes | Core |
| `Synapse.LLM` | Routeur multi-provider | Core |
| `Synapse.Simulation` | Entités sentientes | Core |
| `Synapse.Infrastructure` | Config, logging, qualité | Core |
| `Synapse.Runtime` | EngineHost, scènes, orchestration | Tous |
| `Synapse.Studio` | UI Avalonia + mode `--engine` | Runtime |

## Pipeline G-DNN + L-DNN

```mermaid
flowchart LR
    subgraph Input["Entrées"]
        Mesh[Mesh / Génome]
        Scene[Scène .synapse]
        LLMHints[Hints LLM]
    end

    subgraph GDNN["G-DNN — Géométrie"]
        SDF[SDF neural]
        Poly[Polygonisation LOD]
        RayM[Ray marching BVH]
    end

    subgraph LDNN["L-DNN — Éclairage"]
        GBuf[G-Buffer étendu]
        SSGI[SSGI + cascades]
        AO[SSAO neural]
        Fog[Brouillard froxel]
    end

    subgraph Output["Sortie"]
        Frame[Frame Vulkan]
        GLTF[Export glTF/GLB]
    end

    Mesh --> SDF
    Scene --> SDF
    LLMHints --> LDNN
    SDF --> Poly
    SDF --> RayM
    RayM --> GBuf
    GBuf --> SSGI
    GBuf --> AO
    SSGI --> Frame
    AO --> Frame
    Fog --> Frame
    Poly --> GLTF
```

## Physique : living laws + rigid bodies

```mermaid
flowchart TB
    subgraph Laws["Living Laws"]
        Lib[LivingLawLibrary — 100 lois]
        Comp[LivingLawCompiler]
        Field[PhysicsField]
    end

    subgraph Rigid["Corps rigides"]
        RBW[RigidBodyWorld]
        Joints[PhysicsJoints]
        Vehicle[VehicleController]
        MeshCol[MeshCollision]
    end

    subgraph Orch["MultiphysicsOrchestrator"]
        Tick[Pas fixe dt]
    end

    Lib --> Comp
    Comp --> Field
    RBW --> Joints
    RBW --> Vehicle
    RBW --> MeshCol
    Field --> Tick
    RBW --> Tick
```

## CI/CD

```mermaid
flowchart LR
    Push[Push / PR] --> Build[build.yml]
    Push --> Analysis[analysis.yml]
    Tag[Tag v*] --> Release[release.yml]
    Site[Push site/**] --> Pages[pages.yml]

    Build --> TestLinux[test-linux + Coverlet + Codecov]
    Build --> TestMacOS[test-macos]
    Build --> PubWin[publish-windows]
    Build --> PubLinux[publish-linux]
    Build --> Cert[industrial-certification]

    Analysis --> Analyzers[Roslyn Analyzers]
    Analysis --> Format[dotnet format]

    Release --> Matrix[win/linux/osx artifacts]
    Release --> GHRelease[GitHub Release]
```

## Taille des modules (indicatif)

Le monolithe `PhysicsState.cs` (276 Ko) a été découpé en **26 fichiers** modulaires :

| Fichier | Contenu |
|---|---|
| `PhysicalConstants.cs` | Constantes CODATA 2018 |
| `PhysicsEnums.cs` | Énumérations (FieldLayer, UnitSystem, etc.) |
| `Vector3D.cs`, `Tensor3D.cs`, `QuaternionD.cs` | Algèbre 3D |
| `GeometryPrimitives.cs` | AABB, Ray, Plane, Frustum |
| `AutoDiff.cs` | Différentiation automatique forward-mode |
| `PhysicsStateCore.cs` | Struct PhysicsState (256 octets) + opérateurs |
| `UnitConverter.cs`, `MaterialDatabase.cs` | Conversions et matériaux |
| `OctreeNode.cs`, `KdTree.cs`, `GridHash.cs` | Structures spatiales |

Autres gros modules (post-découpage v1.3) :

| Module | Fichiers | Contenu principal |
|---|---|---|
| `NeatGEvolutionEngine.*.cs` | 65 | Évolution NEAT-G (découpé depuis le monolithe) |
| `VulkanRhiDevice.*.cs` | 9 | Device Vulkan (découpé) |
| `Solvers.*.cs` | 56 | Solveurs numériques (découpé) |
| `LivingLawCompiler.cs` | ~294 KB | Compilateur de lois — **prochain candidat au découpage** |

Voir [CONTRIBUTING.md](../CONTRIBUTING.md) pour les conventions de contribution.

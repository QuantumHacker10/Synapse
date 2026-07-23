# Guide de démarrage rapide — Synapse OMNIA

Ce guide vous permet de lancer Synapse en **moins de 10 minutes**. Pour la documentation complète, voir [README.md](README.md) et [docs/API.md](docs/API.md).

## Prérequis

| Composant | Détail |
|---|---|
| .NET SDK | **10.0.300** (verrouillé dans [`global.json`](global.json)) |
| GPU | Pilote **Vulkan** à jour |
| OS | Windows, Linux ou macOS (arm64) |

Installez le SDK : https://dotnet.microsoft.com/download/dotnet/10.0

```bash
dotnet --version   # doit afficher 10.0.300 ou compatible
```

## 1. Cloner et compiler

```bash
git clone https://github.com/QuantumHacker10/Synapse.git
cd Synapse
dotnet build
```

## 2. Lancer les tests

```bash
dotnet test
# Attendu : suite xUnit verte (voir badge Tests du README)
```

Avec couverture de code (comme en CI) :

```bash
dotnet test --collect:"XPlat Code Coverage"
```

## 3. Lancer Synapse Studio

### Interface graphique (Avalonia)

```bash
dotnet run --project src/Synapse.Studio
```

### Mode moteur seul (GLFW, sans UI)

```bash
dotnet run --project src/Synapse.Studio -- --engine
```

### Charger la scène d'exemple

```bash
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse
```

## 4. Options CLI utiles

| Option | Description | Exemple |
|---|---|---|
| `--width` / `--height` | Résolution viewport | `--width 1920 --height 1080` |
| `--scene` | Chemin vers un projet `.synapse` | `--scene samples/demo.synapse` |
| `--quality` | Preset qualité (`Tiny`, `Small`, `Full`) | `--quality Small` |
| `--engine` / `--glfw` | Mode moteur GLFW sans UI Studio | `--engine` |
| `--validation` | Activer la validation stricte | `--validation` |

Variables d'environnement équivalentes : `SYNAPSE_WIDTH`, `SYNAPSE_HEIGHT`, `SYNAPSE_SCENE`.

## 5. Créer une scène `.synapse`

Les projets Synapse sont des fichiers JSON. Exemple minimal :

```json
{
  "name": "Ma première scène",
  "version": "1.0",
  "activeLawId": "heat_equation",
  "entities": [
    {
      "id": "11111111-1111-1111-1111-111111111111",
      "name": "Sol",
      "type": "Mesh",
      "position": { "x": 0, "y": 0, "z": 0 },
      "scale": { "x": 10, "y": 0.2, "z": 10 },
      "visible": true
    }
  ],
  "camera": {
    "position": { "x": 0, "y": 2, "z": 5 },
    "yaw": -90,
    "pitch": 0,
    "fov": 60
  },
  "assets": {}
}
```

Sauvegardez-le (ex. `scenes/ma-scene.synapse`) puis :

```bash
dotnet run --project src/Synapse.Studio -- --scene scenes/ma-scene.synapse
```

Voir [`samples/demo.synapse`](samples/demo.synapse) pour une scène complète avec entités, génome et agents.

## 6. Utiliser le moteur en code C#

Exemple minimal pour intégrer le runtime dans votre propre application :

```csharp
using Synapse.Infrastructure.Configuration;
using Synapse.Infrastructure.Logging;
using Synapse.Runtime;

var config = SynapseConfig.Load();
var logger = SynapseLogger.Default;

await using var host = new EngineHost(config, logger);

// Initialiser physique, LLM, évolution, etc.
host.InitializeModules();

// Rendu GLFW (Linux/macOS/Windows) ou InitializeRenderFromHwnd sur Windows
host.InitializeRender(1280, 720);

// Boucle principale
while (host.IsRenderInitialized)
{
    host.TickPhysics(1.0 / 60.0);
    await host.TickSimulationAsync();
    host.TickRender();
}
```

### Changer une loi physique à chaud

```csharp
// Après InitializeModules() — la loi par défaut est "heat_equation"
// Lister les 100 lois disponibles
foreach (var (id, name, expr) in host.ListLaws())
    Console.WriteLine($"{id}: {name}");

// Compiler et activer une loi existante
host.CompileLaw("heat_equation", "dT/dt = alpha * laplacian(T)");

// Ou compiler une expression personnalisée
var result = host.CompileLaw("ma_loi", "dT/dt = 0.1 * laplacian(T)");
if (result.Success)
    Console.WriteLine($"Loi active ({result.InstructionCount} ops)");
```

### Interroger le routeur LLM

```csharp
using GDNN.Llm;

// Configurer via variables d'environnement (jamais en dur dans le code)
// OPENAI_API_KEY, ANTHROPIC_API_KEY, OLLAMA_HOST, etc.

var router = host.LlmRouter!;
var response = await router.RouteAsync(
    "Décris une scène de forêt brumeuse au crépuscule",
    new PromptContext { TaskType = LlmTaskType.CreativeWriting },
    cancellationToken: default);

Console.WriteLine(response.Content);
```

## 7. Configuration LLM (optionnel)

| Variable | Provider |
|---|---|
| `OLLAMA_HOST` | Ollama local (ex. `http://localhost:11434`) |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI |

Le routeur [`HybridLlmRouter`](src/Synapse.LLM/HybridLlmRouter.cs) choisit automatiquement le provider disponible selon coût, latence et confidentialité.

Fichier de config : [`src/Synapse.Studio/appsettings.json`](src/Synapse.Studio/appsettings.json).

## 8. Publier un binaire

### Windows x64

```bash
dotnet publish src/Synapse.Studio/Synapse.Studio.csproj \
  -c Release -r win-x64 --self-contained true \
  -o artifacts/Synapse-win-x64
```

Placez `glfw3.dll` (GLFW 3.4+) à côté de l'exécutable. Voir [`src/Synapse.Studio/native/README.md`](src/Synapse.Studio/native/README.md).

### Toutes les plateformes

```bash
bash scripts/publish-all.sh
# Produit win-x64, linux-x64 et osx-arm64 dans artifacts/
```

## 9. Dépannage

| Problème | Solution |
|---|---|
| `Vulkan not found` | Mettez à jour les pilotes GPU ; sur macOS, MoltenVK est requis |
| `glfw3.dll missing` | Copiez la DLL native (voir `src/Synapse.Studio/native/`) |
| Tests échouent sur CI locale | Vérifiez .NET 10.0.300 : `dotnet --version` |
| LLM ne répond pas | Vérifiez `OLLAMA_HOST` ou les clés API ; le routeur bascule automatiquement |
| Viewport noir (Windows) | Vérifiez le pilote Vulkan ; essayez `--engine` pour isoler le rendu |

## 10. Prochaines étapes

- [docs/API.md](docs/API.md) — APIs publiques par module
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diagrammes d'architecture
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribuer au projet
- [CHANGELOG.md](CHANGELOG.md) — historique des versions
- [Issues GitHub](https://github.com/QuantumHacker10/Synapse/issues) — bugs et idées

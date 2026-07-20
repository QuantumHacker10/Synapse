# Third-Party Notices — Synapse OMNIA

Ce fichier recense les dépendances NuGet externes et leurs licences.
Généré manuellement à partir des fichiers `.csproj` — vérifier périodiquement
avec `dotnet list package --include-transitive`.

Dernière vérification : 2026-07-20

## Dépendances directes (runtime)

| Package | Version | Licence | Usage |
|---|---|---|---|
| [Avalonia](https://www.nuget.org/packages/Avalonia) | 11.2.3 | MIT | UI framework (Synapse Studio) |
| [Avalonia.Desktop](https://www.nuget.org/packages/Avalonia.Desktop) | 11.2.3 | MIT | Backend desktop Avalonia |
| [Avalonia.Themes.Fluent](https://www.nuget.org/packages/Avalonia.Themes.Fluent) | 11.2.3 | MIT | Thème Fluent UI |
| [Avalonia.Fonts.Inter](https://www.nuget.org/packages/Avalonia.Fonts.Inter) | 11.2.3 | MIT | Police Inter |
| [CommunityToolkit.Mvvm](https://www.nuget.org/packages/CommunityToolkit.Mvvm) | 8.* | MIT | MVVM helpers |
| [Tmds.DBus.Protocol](https://www.nuget.org/packages/Tmds.DBus.Protocol) | 0.21.3 | MIT | DBus (Linux, override sécurité) |

## Dépendances directes (build / analyse)

| Package | Version | Licence | Usage |
|---|---|---|---|
| [Microsoft.CodeAnalysis.NetAnalyzers](https://www.nuget.org/packages/Microsoft.CodeAnalysis.NetAnalyzers) | 10.* | MIT | Analyseurs Roslyn (Directory.Build.props) |

## Dépendances directes (tests)

| Package | Version | Licence | Usage |
|---|---|---|---|
| [Microsoft.NET.Test.Sdk](https://www.nuget.org/packages/Microsoft.NET.Test.Sdk) | 17.* | MIT | SDK de test |
| [xunit](https://www.nuget.org/packages/xunit) | 2.* | Apache-2.0 | Framework de test |
| [xunit.runner.visualstudio](https://www.nuget.org/packages/xunit.runner.visualstudio) | 2.* | Apache-2.0 | Runner Visual Studio |
| [FluentAssertions](https://www.nuget.org/packages/FluentAssertions) | 6.* | Apache-2.0 | Assertions fluides |
| [coverlet.collector](https://www.nuget.org/packages/coverlet.collector) | 6.* | MIT | Couverture de code |

## Dépendances natives (non NuGet)

| Composant | Version | Licence | Usage |
|---|---|---|---|
| [GLFW](https://www.glfw.org/) | 3.4+ | zlib/libpng | Fenêtrage natif (Linux/macOS/Windows) |
| [Vulkan SDK](https://vulkan.lunarg.com/) | — | Apache-2.0 / divers | Rendu GPU |
| [MoltenVK](https://github.com/KhronosGroup/MoltenVK) | — | Apache-2.0 | Vulkan sur macOS |
| [DXC / glslang](https://github.com/microsoft/DirectXShaderCompiler) | — | LLVM (Apache-2.0 avec exception LLVM) | Compilation shaders → SPIR-V |

## Dépendances transitives notables (Avalonia)

Les packages Avalonia tirent transitivement notamment :

| Package | Licence |
|---|---|
| SkiaSharp | MIT |
| HarfBuzzSharp | MIT |
| MicroCom.Runtime | MIT |
| System.IO.Pipelines | MIT |

Pour la liste complète à jour :

```bash
dotnet list src/Synapse.Studio/Synapse.Studio.csproj package --include-transitive
dotnet list tests/Synapse.Tests/Synapse.Tests.csproj package --include-transitive
```

## Compatibilité des licences

Toutes les dépendances NuGet directes sont sous licences permissives (MIT ou Apache-2.0),
compatibles avec la licence MIT du projet Synapse OMNIA.

## Vérification automatisée

La CI exécute `dotnet restore` et les analyseurs de sécurité NuGet (NU1903+).
Pour auditer les vulnérabilités connues :

```bash
dotnet list package --vulnerable --include-transitive
```

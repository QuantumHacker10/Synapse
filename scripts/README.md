# Scripts — Synapse OMNIA

Utilitaires de maintenance du dépôt.

| Script | Usage |
|---|---|
| `split-monoliths.py` | Découpe un gros fichier C# en modules (`#region` ou types top-level). Cible actuelle : `LivingLawCompiler.cs`. |
| `publish-all.sh` | Publish Studio pour win/linux/osx (voir README). |
| `verify-licenses.sh` | Vérifie les licences NuGet (CI). |

## Découpage monolithe

Les monolithes v1.3 (`NeatGEvolutionEngine`, `VulkanRhiDevice`, `Solvers`) sont déjà découpés.
Pour continuer le découpage de `LivingLawCompiler.cs` :

```bash
python scripts/split-monoliths.py
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format Synapse.slnx whitespace
```

Les scripts PowerShell `split-monoliths.ps1` et `split-vulkan.ps1` ont été retirés (remplacés par le script Python unique).

# Contribuer à Synapse

Merci de votre intérêt pour **Synapse OMNIA**, moteur de simulation 3D. Ce guide décrit le flux Git, les branches protégées et le versionnement.

## Branches

| Branche | Rôle | Qui peut pousser |
|---|---|---|
| `main` | Release stable, tags `v*` | Pull requests uniquement (branche protégée) |
| `develop` | Intégration continue des features | Pull requests depuis `feat/*`, `fix/*`, etc. |
| `feat/*` | Nouvelle fonctionnalité | Développeur / équipe |
| `fix/*` | Correction de bug | Développeur / équipe |
| `docs/*` | Documentation seule | Développeur / équipe |
| `chore/*` | CI, tooling, dépendances | Développeur / équipe |

### Flux recommandé

```text
feat/ma-fonctionnalite ──PR──► develop ──PR──► main ──tag──► v1.2.0
```

1. Partir de `develop` à jour : `git checkout develop && git pull`
2. Créer une branche isolée : `git checkout -b feat/ma-fonctionnalite`
3. Commiter des changements atomiques avec des messages clairs
4. Vérifier localement : `dotnet build && dotnet test`
5. Pousser et ouvrir une PR vers `develop`
6. Après revue et CI verte, merger dans `develop`
7. Pour une release : PR `develop` → `main`, puis taguer (voir ci-dessous)

### Conventions de nommage

```text
feat/neural-shadows      # nouvelle capacité
fix/vulkan-resize-crash  # correction
docs/changelog-1.2       # documentation
chore/ci-cache           # infrastructure
```

## Branche `main` protégée

La branche `main` doit rester stable. Configuration cible :

- **Pull requests obligatoires** — pas de push direct
- **1 revue** minimum avant merge
- **CI verte** — checks requis :
  - `test-linux`
  - `publish-windows`
  - `analyze`
- **Branches à jour** — rebase ou merge de `main` avant merge de la PR
- **Pas de force-push** sur `main`

### Appliquer la protection (GitHub)

> **Note :** la protection de branche via l'API GitHub nécessite un dépôt **public** ou un plan **GitHub Pro** (Team/Enterprise). Sur un dépôt privé gratuit, configurez manuellement dans **Settings → Branches** ou rendez le dépôt public.

Script automatique (une fois éligible) :

```powershell
.\.github\scripts\apply-branch-protection.ps1
```

Configuration manuelle : **Settings → Branches → Add branch protection rule** → pattern `main`, puis activer les options ci-dessus et sélectionner les trois checks CI.

## Versionnement et tags

Le projet suit [Semantic Versioning](https://semver.org/) :

| Composant | Quand l'incrémenter |
|---|---|
| **MAJOR** (`2.0.0`) | Changement incompatible de l'API ou du format de projet |
| **MINOR** (`1.2.0`) | Nouvelle fonctionnalité rétro-compatible |
| **PATCH** (`1.1.1`) | Correction de bug rétro-compatible |

### Créer une release

1. Mettre à jour [CHANGELOG.md](CHANGELOG.md) (section `[Non publié]` → `[X.Y.Z] — date`)
2. Merger `develop` → `main` via PR
3. Taguer et pousser :

```bash
git checkout main
git pull
git tag -a v1.2.0 -m "Synapse OMNIA 1.2.0"
git push origin v1.2.0
```

Le workflow [`.github/workflows/release.yml`](.github/workflows/release.yml) publie automatiquement le zip Windows x64 sur GitHub Releases.

### Tags existants

| Tag | Description |
|---|---|
| `v1.1.0` | Release initiale Synapse OMNIA 1.1 |

## CHANGELOG

Chaque PR significative doit mettre à jour [CHANGELOG.md](CHANGELOG.md) sous la section `[Non publié]`, en catégories :

- **Ajouté** — nouvelles fonctionnalités
- **Modifié** — changements de comportement existant
- **Corrigé** — corrections de bugs
- **Supprimé** — fonctionnalités retirées
- **Sécurité** — correctifs de vulnérabilité

## Qualité

```bash
dotnet build
dotnet test
dotnet format --verify-no-changes   # comme en CI
```

Les workflows [`build.yml`](.github/workflows/build.yml) et [`analysis.yml`](.github/workflows/analysis.yml) doivent passer avant merge.

## Licence et contributions

Ce projet est sous **licence propriétaire** ([LICENSE](LICENSE)). En contribuant, vous acceptez que :

- vous ne copiez pas le code vers d'autres dépôts sans autorisation ;
- vos contributions acceptées peuvent être intégrées sous la licence du projet ;
- vous ne revendiquez pas la paternité du code existant.

Les forks GitHub sont **interdits par la licence** ([LICENSE](LICENSE)). Sur un dépôt public personnel,
GitHub peut toutefois afficher le bouton « Fork » — cela ne constitue pas une autorisation légale ;
toute republication non autorisée reste une violation des droits d'auteur.

## Questions

Ouvrez une [issue](https://github.com/QuantumHacker10/Synapse/issues) ou une [discussion](https://github.com/QuantumHacker10/Synapse/discussions) pour bugs, idées ou questions d'architecture.

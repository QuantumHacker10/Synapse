# Contribuer à Synapse

Merci de votre intérêt pour **Synapse OMNIA**, outil de simulation 3D. Ce guide décrit le flux Git, les branches protégées et le versionnement.

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
feat/ma-fonctionnalite ──PR──► develop ──PR──► main ──tag──► v2.6.0
```

1. Partir de `develop` à jour : `git checkout develop && git pull`
2. Créer une branche isolée : `git checkout -b feat/ma-fonctionnalite`
3. Commiter des changements atomiques avec des messages clairs
4. Vérifier localement : `dotnet build && dotnet test && dotnet format whitespace --verify-no-changes`
5. Pousser et ouvrir une PR vers `develop`
6. Après revue et CI verte, merger dans `develop`
7. Pour une release : PR `develop` → `main`, puis taguer (voir ci-dessous)

### Conventions de nommage

```text
feat/neural-shadows      # nouvelle capacité
fix/vulkan-resize-crash  # correction
docs/changelog-2.4       # documentation
chore/ci-cache           # infrastructure
```

## Branche `main` protégée

La branche `main` doit rester stable. Configuration cible :

- **Pull requests obligatoires** — pas de push direct
- **1 revue** minimum avant merge
- **CI verte** — checks requis :
  - `test-linux`
  - `test-macos`
  - `publish-smoke`
  - `analyze`
  - `CodeQL` (optionnel mais recommandé)
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
git tag -a v2.6.0 -m "Synapse OMNIA 2.6.0"
git push origin v2.6.0
```

Le workflow [`.github/workflows/release.yml`](.github/workflows/release.yml) publie automatiquement les artefacts multi-RID sur GitHub Releases.

### Tags existants

| Tag | Description |
|---|---|
| `v2.9.0` | UDIM/MDL, blend shapes, texture stream, marketplace distant (voir docs/PRODUCTION.md) |
| `v2.8.0` | UsdSkel SkelAnimation clips |
| `v2.7.0` | OpenUSD PBR textures avancées |
| `v2.6.0` | STUN/TURN + OpenUSD materials/skel/variants |
| `v2.5.0` | Production VR / WAN / OpenUSD DCC |
| `v2.4.0` | Production-ready desktop |
| `v2.3.x` | Multi-plateforme mid-range + USDC + blueprints live |

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
dotnet format whitespace --verify-no-changes   # comme en CI analysis.yml
dotnet run --project src/Synapse.Studio -- --health
```

Les workflows [`build.yml`](.github/workflows/build.yml) et [`analysis.yml`](.github/workflows/analysis.yml) doivent passer avant merge.
Voir aussi [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Licence et contributions

Ce projet est sous **licence MIT** ([LICENSE](LICENSE)). En contribuant, vous acceptez que :

- vos contributions acceptées sont intégrées sous la licence MIT du projet ;
- vous êtes l'auteur original de ce que vous soumettez et ne violez pas les droits de tiers ;
- vous conservez vos droits d'auteur sur vos contributions, tout en accordant les droits
  prévus par la licence MIT au projet et à ses utilisateurs.

Les forks et dérivés sont autorisés sous les conditions de la licence MIT.

## Questions

Ouvrez une [issue](https://github.com/QuantumHacker10/Synapse/issues) ou une [discussion](https://github.com/QuantumHacker10/Synapse/discussions) pour bugs, idées ou questions d'architecture.

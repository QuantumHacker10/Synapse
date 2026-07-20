# Contributing to Synapse

Thank you for your interest in **Synapse OMNIA**, a 3D simulation engine. This guide
describes the Git workflow, protected branches, and versioning.

## Branches

| Branch | Role | Who can push |
|---|---|---|
| `main` | Stable release, `v*` tags | Pull requests only (protected branch) |
| `develop` | Continuous feature integration | Pull requests from `feat/*`, `fix/*`, etc. |
| `feat/*` | New feature | Developer / team |
| `fix/*` | Bug fix | Developer / team |
| `docs/*` | Documentation only | Developer / team |
| `chore/*` | CI, tooling, dependencies | Developer / team |

### Recommended flow

```text
feat/my-feature ──PR──► develop ──PR──► main ──tag──► v1.2.0
```

1. Start from an up-to-date `develop`: `git checkout develop && git pull`
2. Create an isolated branch: `git checkout -b feat/my-feature`
3. Commit atomic changes with clear messages
4. Verify locally: `dotnet build && dotnet test`
5. Push and open a PR toward `develop`
6. After review and green CI, merge into `develop`
7. For a release: PR `develop` → `main`, then tag (see below)

### Naming conventions

```text
feat/neural-shadows      # new capability
fix/vulkan-resize-crash  # bug fix
docs/changelog-1.2       # documentation
chore/ci-cache           # infrastructure
```

## Protected `main` branch

The `main` branch must stay stable. Target configuration:

- **Pull requests required** — no direct push
- **1 review** minimum before merge
- **Green CI** — required checks:
  - `test-linux`
  - `publish-windows`
  - `analyze`
- **Up-to-date branches** — rebase or merge `main` before merging the PR
- **No force-push** on `main`

### Apply protection (GitHub)

> **Note:** branch protection via the GitHub API requires a **public** repository or a
> **GitHub Pro** plan (Team/Enterprise). On a free private repo, configure manually in
> **Settings → Branches** or make the repository public.

Automatic script (once eligible):

```powershell
.\.github\scripts\apply-branch-protection.ps1
```

Manual configuration: **Settings → Branches → Add branch protection rule** → pattern `main`,
then enable the options above and select the three CI checks.

## Versioning and tags

The project follows [Semantic Versioning](https://semver.org/):

| Component | When to increment |
|---|---|
| **MAJOR** (`2.0.0`) | Incompatible API or project format change |
| **MINOR** (`1.2.0`) | Backward-compatible new feature |
| **PATCH** (`1.1.1`) | Backward-compatible bug fix |

### Create a release

1. Update [CHANGELOG.md](CHANGELOG.md) (`[Unreleased]` → `[X.Y.Z] — date`)
2. Merge `develop` → `main` via PR
3. Tag and push:

```bash
git checkout main
git pull
git tag -a v1.2.0 -m "Synapse OMNIA 1.2.0"
git push origin v1.2.0
```

Or use the helper script:

```bash
bash .github/scripts/create-release-tag.sh v1.2.0 "Synapse OMNIA 1.2.0"
```

The [`.github/workflows/release.yml`](.github/workflows/release.yml) workflow automatically
publishes Windows, Linux, and macOS archives on GitHub Releases.

### Existing tags

| Tag | Description |
|---|---|
| [`v1.1.0`](https://github.com/QuantumHacker10/Synapse/releases/tag/v1.1.0) | Initial Synapse OMNIA 1.1 release |
| [`v1.2.0`](https://github.com/QuantumHacker10/Synapse/releases/tag/v1.2.0) | Industrial Synapse OMNIA 1.2 release |

## CHANGELOG

Each significant PR should update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]`, in categories:

- **Added** — new features
- **Changed** — changes to existing behavior
- **Fixed** — bug fixes
- **Removed** — removed features
- **Security** — vulnerability fixes

## Quality

```bash
dotnet build
dotnet test
dotnet format --verify-no-changes   # same as CI
```

The [`build.yml`](.github/workflows/build.yml) and [`analysis.yml`](.github/workflows/analysis.yml)
workflows must pass before merge.

### Codecov (maintainers)

Add repository secret `CODECOV_TOKEN` from [codecov.io](https://codecov.io/gh/QuantumHacker10/Synapse)
so the dynamic coverage badge in README updates after each CI run.

## License and contributions

This project is under the **MIT License** ([LICENSE](LICENSE)). By contributing, you agree that:

- your contribution is your original work;
- you license your contribution under the MIT License;
- you do not violate third-party rights.

Forks and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Questions

Open an [issue](https://github.com/QuantumHacker10/Synapse/issues) or a
[discussion](https://github.com/QuantumHacker10/Synapse/discussions) for bugs, ideas, or
architecture questions.

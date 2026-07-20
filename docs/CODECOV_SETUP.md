# Codecov setup

Synapse uploads test coverage from [`.github/workflows/build.yml`](../.github/workflows/build.yml)
to [Codecov](https://codecov.io/gh/QuantumHacker10/Synapse). The README badge updates after
the first successful upload.

## Option A — OIDC (recommended, no secret)

The workflow uses **OIDC** (`use_oidc: true`) so you do not need `CODECOV_TOKEN` when:

1. The repository is **public**
2. The [Codecov GitHub App](https://github.com/apps/codecov) is installed on `QuantumHacker10/Synapse`
3. The repo is activated at [app.codecov.io](https://app.codecov.io/gh/QuantumHacker10/Synapse)

### One-time steps

1. Sign in at [codecov.io](https://codecov.io) with GitHub
2. Add repository **QuantumHacker10/Synapse**
3. Install the Codecov GitHub App when prompted
4. Push to `main` or merge a PR — CI uploads coverage automatically

## Option B — `CODECOV_TOKEN` secret (fallback)

If OIDC upload fails (private fork, app not installed, org policy), use a repository upload token:

### 1. Get the token

Open [Codecov repo settings](https://app.codecov.io/gh/QuantumHacker10/Synapse/settings) → **General** → copy the **Repository Upload Token** (UUID format, not `CODECOV_TOKEN=...`).

### 2. Add the GitHub secret

**GitHub UI:** `QuantumHacker10/Synapse` → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

| Name | Value |
|---|---|
| `CODECOV_TOKEN` | paste token only (no prefix) |

**CLI (maintainer with `gh` admin access):**

```bash
bash scripts/setup-codecov-secret.sh
# or:
gh secret set CODECOV_TOKEN --repo QuantumHacker10/Synapse
```

### 3. Switch workflow to token mode

In `.github/workflows/build.yml`, replace the Codecov step with:

```yaml
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        files: coverage/report/Cobertura.xml
        flags: unittests
        name: synapse-linux
        fail_ci_if_error: false
```

Remove `use_oidc: true` when using the token.

## Verify

After the next green `Build & Test` run on `main`:

- [Codecov dashboard](https://app.codecov.io/gh/QuantumHacker10/Synapse) shows a report
- README badge `codecov.io/gh/QuantumHacker10/Synapse/graph/badge.svg` shows a percentage

Local coverage (without Codecov):

```bash
dotnet test Synapse.slnx -c Release --collect:"XPlat Code Coverage" --results-directory coverage
dotnet tool install -g dotnet-reportgenerator-globaltool
reportgenerator -reports:'coverage/**/coverage.cobertura.xml' -targetdir:'coverage/report' -reporttypes:TextSummary
cat coverage/report/Summary.txt
```

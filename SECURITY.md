# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| 1.2.x   | :white_check_mark: |
| 1.1.x   | :x:                |
| < 1.1   | :x:                |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, report via one of these channels:

1. **GitHub Security Advisories** (preferred) :
   [Report a vulnerability](https://github.com/QuantumHacker10/Synapse/security/advisories/new)

2. **Private issue** : open a GitHub issue with the title `[SECURITY]` —
   a maintainer will mark it as sensitive.

### What to include

- Description of the vulnerability
- Steps to reproduce
- Affected versions and components
- Potential impact
- Suggested fix (if any)

### Response timeline

| Step | Target |
|---|---|
| Acknowledgement | 48 hours |
| Initial assessment | 7 days |
| Fix or mitigation plan | 30 days |
| Public disclosure | After patch release |

## Security measures in place

- **CodeQL** — static analysis on every push/PR (`.github/workflows/codeql.yml`)
- **Dependabot** — automated dependency updates (`.github/dependabot.yml`)
- **NuGet audit** — `dotnet list package --vulnerable` in CI
- **Branch protection** — PR required on `main`, CI checks mandatory
- **Secret scanning** — API keys never committed (env vars only for LLM)

## Dependency audit

Run locally :

```bash
dotnet list package --vulnerable --include-transitive
```

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license inventory.

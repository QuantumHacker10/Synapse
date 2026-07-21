# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 2.4.x   | :white_check_mark: |
| 2.3.x   | :white_check_mark: |
| &lt; 2.3  | :x: |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report only via **GitHub Security Advisories** (preferred):

[Report a vulnerability](https://github.com/QuantumHacker10/Synapse/security/advisories/new)

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

## Production security notes

- Plugin loading is confined to `--plugin-dir` (path jail). Prefer a `plugins.allow` SHA-256 allowlist for production installs.
- Do not expose `publicBind` P2P without a session auth key (`PeerEncryption` / `--wan-code` for the WAN QA hub).
- See [docs/PRODUCTION.md](docs/PRODUCTION.md) for the production / experimental matrix.

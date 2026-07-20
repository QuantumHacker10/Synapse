# License options for Synapse OMNIA

Synapse OMNIA currently ships under a **proprietary license** ([LICENSE](../LICENSE)).
This document clarifies permitted use today and outlines open-source alternatives the
maintainers may consider in the future.

## Current license (proprietary)

| Permitted without written agreement | Requires written permission |
|---|---|
| Viewing the public source for personal evaluation | Copying or mirroring the repository |
| Running official [Releases](https://github.com/QuantumHacker10/Synapse/releases) binaries | Forking or republishing the source |
| **Academic and non-commercial research** (experiments, papers, coursework — no redistribution of source or binaries) | Commercial products, SaaS, or paid training |
| Filing GitHub issues and authorized pull requests | Derivative works or component extraction |

### Academic use

University students, researchers, and educators may:

- Clone the repository locally for **study and non-commercial experimentation**
- Run Synapse Studio and cite Synapse in academic publications
- Propose improvements via pull requests (subject to the contribution terms in LICENSE)

They may **not**, without permission:

- Redistribute source or release binaries (including course material mirrors)
- Publish forked or extracted components as standalone projects
- Use Synapse in commercial spin-offs or funded products

For research collaborations or institutional licenses, open a
[license request issue](https://github.com/QuantumHacker10/Synapse/issues/new?template=license_request.yml).

## Why not MIT / Apache today?

The proprietary license protects novel pipelines (G-DNN, L-DNN, living laws, NEAT-G
integration) during early product development. Restrictions may be relaxed once business
and community goals are aligned.

## Open-source alternatives under consideration

| License | Pros | Cons |
|---|---|---|
| **MIT** | Maximum adoption; simple contribution story | Minimal patent protection |
| **Apache 2.0** | Patent grant; corporate-friendly | Slightly more legal overhead |
| **Dual license** | Core open + commercial add-ons | Requires clear module boundaries |

A future move to open source would likely:

1. Announce intent in CHANGELOG and README
2. Retag or branch from a known `v*` release
3. Update CONTRIBUTING to welcome public forks
4. Keep trademarks (Synapse OMNIA, Synapse Studio) protected

## Contact

- [License request issue template](https://github.com/QuantumHacker10/Synapse/issues/new?template=license_request.yml)
- [General issues](https://github.com/QuantumHacker10/Synapse/issues)

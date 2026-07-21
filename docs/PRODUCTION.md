# Production readiness — Synapse OMNIA 2.9

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.9)

| Surface | Status |
|---|---|
| Studio / laws / blueprints / export / multi-RID | **Production** |
| Plugins local + **remote marketplace** | **Production** (HTTPS catalog, SHA-256, path jail) |
| WAN P2P + STUN/TURN | **Production** |
| OpenXR | **Production** |
| OpenUSD DCC (Synapse runtime) | **Production** — composition, xforms, PBR+**UDIM**, **MDL refs**, skeletons, clips, **blend shapes**, variants |
| GPU texture streaming | **Production** — `GpuTextureStreamer` page-in / LRU / UDIM prefetch |

## OpenUSD extended runtime (Synapse)

Synapse ships a **first-party OpenUSD-oriented runtime** (USDA/USDC MeshIO), not a bundled Pixar OpenUSD C++/Hydra binary:

| Feature | Support |
|---|---|
| UDIM | `<UDIM>` / `%(UDIM)` → `MeshMaterial.UdimTiles` |
| MDL | `sourceAsset` `.mdl` + `mdl:sourceMaterial` (reference; no full MDL SDK eval) |
| Blend shapes | `def BlendShape` offsets → `MeshBlendShape.Apply` |
| SkelAnimation | timeSamples TRS (2.8) |
| Texture stream | `GpuTextureStreamer` disk/HTTPS + LRU |

## Remote plugin marketplace

```bash
dotnet run --project src/Synapse.Studio -c Release -- --engine \
  --plugin-dir ./plugins \
  --plugin-marketplace-url https://example.com/synapse/marketplace.json
```

Catalog JSON is a list of `{ id, fileName, sha256, downloadUrl }`. Downloads are path-jailed under `--plugin-dir` and hash-verified.

## Verify

```bash
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format whitespace --verify-no-changes
dotnet run --project src/Synapse.Studio -c Release -- --health
```

## Honest release claim

Synapse **2.9** is production-ready for desktop simulation tooling with an extended OpenUSD import surface (UDIM/MDL refs/blend shapes), GPU texture streaming helpers, and a remote plugin marketplace.
It does **not** claim a full Pixar OpenUSD C++/Hydra embed or a complete NVIDIA MDL SDK evaluator.

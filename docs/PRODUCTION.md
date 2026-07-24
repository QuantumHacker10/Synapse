# Production readiness — Synapse OMNIA 2.10

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.10)

| Surface | Status |
|---|---|
| Studio / laws / blueprints / export / multi-RID | **Production** |
| Plugins local + **remote marketplace** | **Production** (HTTPS catalog, SHA-256, path jail) |
| WAN P2P + STUN/TURN | **Production** |
| OpenXR | **Production** |
| OpenUSD DCC (Synapse MeshIO) | **Production** — DCC topology (`faceVertexCounts`), authored normals, multi-mesh, purpose/visibility, extent/doubleSided, PBR+UDIM, MDL refs, skeletons/clips/blend shapes, variants, composition (+ payload mute), diagnostics + `--health` smoke |
| GPU texture streaming | **Production** — `GpuTextureStreamer` page-in / LRU / UDIM prefetch |

## OpenUSD production MeshIO (Synapse)

Synapse ships a **first-party OpenUSD-oriented MeshIO** (USDA + Synapse USDC mesh-pack), not a bundled Pixar OpenUSD C++/Hydra binary:

| Feature | Support |
|---|---|
| Topology | `faceVertexCounts` + indices triangulation; `-1` sentinels; flat tri lists |
| Normals | `normal3f[]` / primvars; smooth generate fallback |
| Multi-mesh | All `def Mesh` prims → `MeshPrimitive`s |
| Purpose / visibility | Skip `guide`/`proxy` / `invisible` (configurable mask) |
| Bounds / sidedness | `extent`, `doubleSided` |
| Materials | UsdPreviewSurface + UsdUVTexture + opacityThreshold / emissiveIntensity / colorSpace |
| UDIM | `<UDIM>` / `%(UDIM)` on albedo and other slots → `UdimTiles` / `UdimMapsBySlot` |
| MDL | `sourceAsset` `.mdl` + `mdl:sourceMaterial` (reference; no full MDL SDK eval) |
| Blend shapes | `def BlendShape` offsets + normalOffsets → `MeshBlendShape.Apply` |
| SkelAnimation | timeSamples TRS + stage `timeCodesPerSecond` |
| Composition | references / payloads (mute) / subLayers / inherits |
| Texture stream | `GpuTextureStreamer` disk/HTTPS + LRU |
| Gate | `UsdProductionSmoke` embedded in `--health` (`usd=ok`) |

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
# expect: … [interactive-ready] usd=ok …
```

## Honest release claim

Synapse **2.10** is production-ready for desktop simulation tooling with a **complete first-party OpenUSD MeshIO import surface** for DCC mesh/material/skel workflows used by Studio.
It does **not** claim a full Pixar OpenUSD C++/Hydra embed, a complete NVIDIA MDL SDK evaluator, or bit-perfect OpenUSD USDC crate fidelity (USDA + Synapse mesh-pack USDC are the supported production paths; foreign USDC remains best-effort).

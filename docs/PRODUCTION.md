# Production readiness — Synapse OMNIA 2.8

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.8)

| Surface | Status |
|---|---|
| Studio edit + living laws + blueprints live | **Production** |
| Headless sim / benchmarks / scene I/O / glTF export | **Production** |
| Multi-RID publish (`win\|linux\|osx` × `x64\|arm64`) | **Production** (CI smoke) |
| Local plugins | **Production** |
| Authenticated LAN/WAN P2P + STUN/TURN | **Production** |
| Vulkan viewport | **Production** when `interactive-ready` |
| OpenXR / VR | **Production** |
| OpenUSD DCC | **Production** — composition, xforms, PBR textures, skeletons, **SkelAnimation clips**, variants |

## UsdSkel animation clips

USDA `def SkelAnimation "Name"` with:

- `token[] joints`
- `float3[] translations` / `quatf[] rotations` / `float3[] scales` (static or `.timeSamples = { t: [...] }`)
- Skeleton link via `rel skel:animationSource = </Name>`

Loaded into `MeshAsset.AnimationClips` (`MeshAnimationClip`). Sample with `Evaluate(time)` / `EvaluateLocalMatrices(time)` (lerp + slerp).

## Verify before shipping

```bash
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format whitespace --verify-no-changes
dotnet run --project src/Synapse.Studio -c Release -- --health
```

## Honest release claim

Synapse **2.8** is production-ready for desktop simulation tooling including UsdSkel animation clip import/sampling.
It does not claim full UsdSkel blend shapes, sparse topology animation, or a complete Pixar USD runtime.

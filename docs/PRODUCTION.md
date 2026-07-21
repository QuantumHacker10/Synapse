# Production readiness — Synapse OMNIA 2.7

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.7)

| Surface | Status |
|---|---|
| Studio edit + living laws + blueprints live | **Production** |
| Headless sim / benchmarks / scene I/O / glTF export | **Production** |
| Multi-RID publish (`win\|linux\|osx` × `x64\|arm64`) | **Production** (CI smoke) |
| Local plugins (`--plugin-dir` + jail + optional allowlist / marketplace) | **Production** |
| Authenticated LAN/WAN P2P + STUN/TURN | **Production** |
| Vulkan viewport (real GPU + GLFW/MoltenVK) | **Production** when health says `interactive-ready` |
| OpenXR / VR | **Production** — native Vulkan2 or labeled simulated |
| OpenUSD DCC import | **Production** — composition, xforms, materials+PBR textures, skeletons, variants |

## OpenUSD textures

USDA `UsdPreviewSurface` inputs may `.connect` to `UsdUVTexture` shaders (`asset inputs:file = @path@`):

| PreviewSurface input | MeshMaterial slot |
|---|---|
| diffuseColor / baseColor | AlbedoTexturePath |
| normal | NormalTexturePath |
| metallic / roughness / occlusionRoughnessMetallic | MetallicRoughnessTexturePath |
| occlusion | AOTexturePath |
| emissiveColor | EmissiveTexturePath |
| displacement / height | HeightTexturePath |
| clearcoat / specularColor / opacity | Clearcoat / Specular / Opacity texture paths |

Relative `@./textures/…@` paths resolve against the USDA directory. Mesh UVs from `primvars:st`.

## WAN + STUN/TURN

See prior 2.6 notes: `--stun-server`, `--turn-server`, `--turn-user`, `--turn-password`.

## Verify before shipping

```bash
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format whitespace --verify-no-changes
dotnet run --project src/Synapse.Studio -c Release -- --health
```

## Honest release claim

Synapse **2.7** is production-ready for desktop simulation tooling including OpenUSD PBR texture maps.
It does not replace a full Pixar USD runtime (UDIM tiles, MDL, GPU texture streaming, or animation clips).

# Production readiness — Synapse OMNIA 2.6

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.6)

| Surface | Status |
|---|---|
| Studio edit + living laws + blueprints live | **Production** |
| Headless sim / benchmarks / scene I/O / glTF export | **Production** |
| Multi-RID publish (`win\|linux\|osx` × `x64\|arm64`) | **Production** (CI smoke) |
| Local plugins (`--plugin-dir` + jail + optional allowlist / marketplace) | **Production** |
| Authenticated LAN/WAN P2P + STUN/TURN | **Production** |
| Vulkan viewport (real GPU + GLFW/MoltenVK) | **Production** when health says `interactive-ready` |
| OpenXR / VR | **Production** — native Vulkan2 or labeled simulated |
| OpenUSD DCC import | **Production** — composition, xforms, materials, skeletons, variants |

## WAN + STUN/TURN

```bash
# Host with STUN (advertise reflexive IP) and optional TURN for symmetric NAT
dotnet run --project src/Synapse.Studio -c Release -- --engine --wan-code myroom \
  --stun-server stun.l.google.com:19302 \
  --turn-server turn.example.com:3478 --turn-user u --turn-password secret

# Peer
dotnet run --project src/Synapse.Studio -c Release -- --engine --wan-code myroom --wan-join \
  --wan-rendezvous 192.168.1.10 --stun-server stun.l.google.com:19302 \
  --turn-server turn.example.com:3478 --turn-user u --turn-password secret
```

Env equivalents: `SYNAPSE_STUN_SERVER`, `SYNAPSE_TURN_SERVER`, `SYNAPSE_TURN_USER`, `SYNAPSE_TURN_PASSWORD`, `SYNAPSE_WAN_PREFER_TURN=1`.

## OpenUSD

Supported on USDA/ASCII `.usd` (composition walk):

- Arcs: references, payloads, subLayers, inherits, `@file@</Prim>`
- Xforms: translate, rotateXYZ, scale, transform, xformOpOrder
- Materials: UsdPreviewSurface (`diffuseColor`, roughness, metallic, opacity) + `material:binding`
- Skeletons: `token[] joints`, `bindTransforms`/`restTransforms`, `primvars:skel:jointIndices/Weights`
- Variants: `variantSet "name" = { "A" {…} "B" {…} }` via `MeshLoadConfig.UsdVariantSelections`

## Verify before shipping

```bash
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format whitespace --verify-no-changes
dotnet run --project src/Synapse.Studio -c Release -- --health
```

## Honest release claim

Synapse **2.6** is production-ready for desktop simulation tooling including STUN/TURN-assisted WAN and
OpenUSD DCC import of meshes with materials, skinning, and variants. It does not replace a full Pixar
USD runtime (schemas beyond the above, crate composition parity, or hosted TURN infra).

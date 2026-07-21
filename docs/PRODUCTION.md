# Production readiness — Synapse OMNIA 2.5

This document defines what **production-ready** means for Synapse today and how to verify it.

## Definition (2.5)

| Surface | Status |
|---|---|
| Studio edit + living laws + blueprints live | **Production** |
| Headless sim / benchmarks / scene I/O / glTF export | **Production** |
| Multi-RID publish (`win\|linux\|osx` × `x64\|arm64`) | **Production** (CI smoke) |
| Local plugins (`--plugin-dir` + jail + optional `plugins.allow` / `marketplace.json`) | **Production** |
| Authenticated LAN/WAN P2P (`MultiPeer` + `PeerEncryption` + NAT rendezvous) | **Production** |
| Vulkan viewport (real GPU + GLFW/MoltenVK) | **Production** when health says `interactive-ready` |
| OpenXR / VR | **Production** — native Vulkan2 swapchain when loader+HMD+bind; otherwise labeled simulated |
| OpenUSD DCC import | **Production** — composition arcs + xformOp TRS/transform stacks |

## WAN CLI

```bash
# Host (opens TCP Any + UDP rendezvous Any:7778)
dotnet run --project src/Synapse.Studio -c Release -- --engine --wan-code myroom --wan-port 7777

# Peer on another machine (rendezvous = host public/LAN IP; port-forward UDP 7778 + TCP 7777 if behind NAT)
dotnet run --project src/Synapse.Studio -c Release -- --engine --wan-code myroom --wan-join \
  --wan-rendezvous 192.168.1.10 --wan-rendezvous-port 7778
```

Symmetric NAT / CGNAT without a reachable rendezvous still needs an externally reachable coordinator (STUN/TURN not bundled).

## OpenXR

- Pass Vulkan instance/device/queue via `OpenXrVulkanBinding` for native `XR_KHR_vulkan_enable2` swapchains.
- Without a headset or loader, the session initializes a **production simulated** swapchain (`IsSimulated=true`) for lifecycle QA.
- Force simulated: `SYNAPSE_VR_FORCE_SIMULATED=1`.

## Verify before shipping

```bash
dotnet build Synapse.slnx -c Release
dotnet test Synapse.slnx -c Release
dotnet format whitespace --verify-no-changes
dotnet run --project src/Synapse.Studio -c Release -- --health
bash scripts/smoke-publish-rids.sh   # optional full 6-RID publish
```

Exit codes for `--health`:
- `0` — core-ready (modules + CPU baseline)
- `2` — not ready

## Security baselines

- Plugins: path jail, UNC/URL block, optional SHA-256 allowlist (`plugins.allow`), local `marketplace.json` verify
- P2P: public bind requires auth; AES-GCM with per-session salt + AAD; HMAC handshake; decrypt-or-drop; MaxPeers; TcpClient dispose
- NAT REGISTER/DISCOVER payloads are MAC-scoped to the session code
- Report vulns via GitHub Security Advisories only — see [SECURITY.md](../SECURITY.md)

## Minimum hardware

See [REQUIREMENTS.md](REQUIREMENTS.md). Mid-range Vulkan 1.1+ GPU + AVX2/NEON CPU baseline. AVX-512 not required.

## Honest release claim

Synapse **2.5** is production-ready for **desktop simulation tooling** including authenticated multi-peer sessions,
OpenUSD composition/xform import, and OpenXR (native or explicitly simulated). It does not bundle STUN/TURN
or claim every OpenUSD schema (materials, skeletons, variants) is implemented.

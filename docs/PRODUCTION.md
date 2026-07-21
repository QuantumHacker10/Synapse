# Production readiness — Synapse OMNIA 2.4

This document defines what **production-ready** means for Synapse today, how to verify it,
and which surfaces remain **experimental**.

## Definition (2.4)

| Surface | Status |
|---|---|
| Studio edit + living laws + blueprints live | **Production** |
| Headless sim / benchmarks / scene I/O / glTF export | **Production** |
| Multi-RID publish (`win\|linux\|osx` × `x64\|arm64`) | **Production** (CI smoke) |
| Local plugins (`--plugin-dir` + jail + optional `plugins.allow` / `marketplace.json`) | **Production** |
| Authenticated LAN P2P (`MultiPeer` + `PeerEncryption`) | **Production** |
| Vulkan viewport (real GPU + GLFW/MoltenVK) | **Production** when health says `interactive-ready` |
| WAN NAT (`--wan-code`) | **Experimental** — loopback rendezvous for QA |
| OpenXR / VR swapchain | **Experimental** — loader detect + simulated image handles |
| Full OpenUSD crate / complex Xform stacks | **Best-effort** — composition refs + translate supported |

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
- Report vulns via GitHub Security Advisories only — see [SECURITY.md](../SECURITY.md)

## Minimum hardware

See [REQUIREMENTS.md](REQUIREMENTS.md). Mid-range Vulkan 1.1+ GPU + AVX2/NEON CPU baseline. AVX-512 not required.

## Honest release claim

Synapse **2.4** is production-ready for **desktop simulation tooling** (edit, simulate, export, plugins, authenticated local peers).
It is **not** a claim of enterprise VR, global WAN mesh, or full OpenUSD DCC parity.

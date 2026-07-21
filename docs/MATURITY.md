# Maturité produit — Synapse OMNIA

Synapse OMNIA **v2.2** est en **accès anticipé / R&D avancée**, pas un produit
« production-ready » pour des charges critiques, de la VR réelle ou du multi-joueurs WAN.

Ce document est la source de vérité des claims. Le catalogue code miroir est
[`FeatureMaturityCatalog`](../src/Synapse.Core/Maturity/FeatureMaturityCatalog.cs).

## Cible officielle

| Élément | Statut |
|---|---|
| Plateforme de référence | **Windows x64 + GPU Vulkan réel** |
| Linux / macOS | Builds natifs possibles ; viewport Studio et validation GPU moins matures |
| Headless CI / cloud sans GPU | Éditeur UI OK ; rendu 3D et ticks live limités (voir `AGENTS.md`) |

## Matrice

| Id | Surface | Tier | Notes |
|---|---|---|---|
| `Studio.Editor` | Synapse Studio | **EarlyAccess** | Édition, inspecteur, console LLM ; viewport Vulkan sur Windows |
| `Runtime.Local` | EngineHost + lois + multiphysique | **EarlyAccess** | Cœur local utilisable ; couverture et durcissement incomplets |
| `Runtime.Benchmarks` | Benchmarks headless | **EarlyAccess** | Régression / perf, pas certification industrielle |
| `IO.GlTF` | Export glTF / import mesh | **EarlyAccess** | FBX ASCII + USDA limités |
| `Plugins.CSharp` | Plugins C# | **EarlyAccess** | ALC isolé ≠ sandbox; `SYNAPSE_PLUGIN_TRUST=require-manifest` |
| `Network.P2P` | P2P multi-pairs | **Experimental** | Labo TCP + framing exact ; pas collaboratif production |
| `Network.WAN` | WAN NAT + AES-GCM | **Experimental** | Rendez-vous **loopback** avec registre de ports ; pas de NAT réel |
| `VR.OpenXR` | OpenXR swapchain | **Experimental** | Fail-closed hors `SYNAPSE_VR_SIMULATE=1` ; handles synthétiques |
| `Web.Editor` | Éditeur web glTF | **Experimental** | Preview site ; pas Studio WASM production |

### Légende des tiers

| Tier | Signification |
|---|---|
| **Supported** | Validé pour un usage local sur la cible officielle (aucun item en v2.2) |
| **EarlyAccess** | Utilisable ; expectez des trous, du churn API et une validation incomplète |
| **Experimental** | Scaffold / stub / localhost — **ne pas** présenter comme capacité production |

## Ce que v2.2 assume honnêtement

- Studio + runtime local pour explorer scènes, lois vivantes, agents et benchmarks.
- Surfaces VR / P2P WAN / web marquées `[SynapseExperimental]` dans le code.
- Pas de promesse de path tracing certifié, FEM production, ou simulation collaborative WAN.

## Prochaines priorités de durcissement

1. Couverture de tests et réduction des warnings analyseurs sur le cœur Runtime/Physics.
2. Scènes d'exemple réalistes sous `samples/` exercées en CI.
3. Promouvoir une surface en **Supported** seulement après validation Windows+GPU réelle.
4. Remplacer les scaffolds VR / WAN par des intégrations natives — ou les retirer des claims produit.

### Durcissement déjà livré (production-hardening)

- GLB : validation des bornes de chunks avant copie.
- `ZeroCopyBuffer` / `MappedBuffer` : mapping fichier réel + durée de vie du handle.
- Streaming : assets manquants en fail-closed ; stage GPU avec intégrité des poids.
- Marketplace `.synapse-law` : validation taille / expression / catalogue.
- Plugins : mode `RequireManifest` + SHA-256 ; ALC documenté hors sandbox.
- P2P : `ReadExactly` sur les frames ; WAN loopback avec registre de ports correct.
- VR : fail-closed sauf `SYNAPSE_VR_SIMULATE=1`.
- Studio : erreurs projet / viewport remontées à l'UI.

Voir aussi [ROADMAP.md](../ROADMAP.md) et [SECURITY.md](../SECURITY.md).

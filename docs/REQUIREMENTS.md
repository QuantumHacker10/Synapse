# Configuration minimale — Synapse OMNIA

Cible d’adoption : **machines milieu de gamme** avec GPU Vulkan et CPU x64/Arm64 courant.
Synapse n’exige **pas** AVX-512, ni Vulkan 1.4, ni Windows uniquement.

## Matériel (minimum recommandé)

| Composant | Minimum | Recommandé |
|---|---|---|
| **CPU** | x64 ou Arm64, SSE2 (Intel/AMD) ou NEON (Apple/Arm) | AVX2 (Intel Haswell+ / AMD Excavator+, ~2013–2015) ou Apple Silicon |
| **RAM** | 8 Go | 16 Go |
| **GPU** | Carte / iGPU avec **Vulkan 1.1+** (pilotes à jour) | Vulkan **1.2+**, 4 Go VRAM |
| **Stockage** | 2 Go libres | SSD |

### Ce qui n’est **pas** requis
- **AVX-512** — optionnel (`SYNAPSE_ALLOW_AVX512=1` ou `SYNAPSE_SIMD_MAX=avx512`)
- **Ray tracing matériel** — désactivé automatiquement si absent
- **GPU discret** — les iGPU Intel UHD / AMD / Apple Metal→MoltenVK sont supportés
- **Windows x64 seul** — Linux x64/arm64 et macOS arm64/x64 sont des cibles natives

## Logiciel

| Composant | Minimum |
|---|---|
| [.NET runtime / SDK](https://dotnet.microsoft.com/download) | **10.0.300** (voir `global.json`) |
| Pilote GPU | Vulkan (NVIDIA / AMD / Intel) ; **MoltenVK** sur macOS |
| Fenêtrage | **GLFW 3.3+** (`glfw3.dll` / `libglfw.so.3` / `libglfw.3.dylib`) |
| OS | Windows 10/11, Linux (glibc récent), macOS 12+ |

## Plateformes natives (publish)

| RID | Statut |
|---|---|
| `win-x64` | Officiel |
| `win-arm64` | Supporté |
| `linux-x64` | Officiel |
| `linux-arm64` | Supporté |
| `osx-arm64` | Officiel (MoltenVK) |
| `osx-x64` | Supporté (MoltenVK) |

```bash
# Exemples
dotnet publish src/Synapse.Studio -c Release -r win-x64 --self-contained
dotnet publish src/Synapse.Studio -c Release -r linux-arm64 --self-contained
dotnet publish src/Synapse.Studio -c Release -r osx-x64 --self-contained
# Ou toutes les RID principales :
./scripts/publish-all.sh
```

## Baseline moteur (compatibilité)

| Domaine | Baseline milieu de gamme |
|---|---|
| **Graphique** | Vulkan **1.2** demandé à l’instance ; extensions avancées activées seulement si présentes |
| **SIMD** | Production = **AVX2 / NEON** ; AVX-512 opt-in |
| **Fenêtre** | GLFW sur tous les OS ; HWND = embed Studio Windows uniquement |
| **Meshes** | glTF/GLB, OBJ, FBX ASCII, USDA, **USDC** (mesh-pack Synapse + best-effort OpenUSD) |

## Variables d’environnement utiles

| Variable | Effet |
|---|---|
| `SYNAPSE_SIMD_MAX` | `scalar` \| `sse2` \| `avx2` \| `avx512` \| `auto` (défaut) |
| `SYNAPSE_ALLOW_AVX512` | `1` / `true` pour autoriser AVX-512 en mode `auto` |
| `SYNAPSE_SEED` | Graine de reproductibilité simulation |
| `VULKAN_SDK` | Chemin SDK Vulkan (Linux/macOS) pour le loader |

## Vérifier sa machine

Au démarrage, `NativePlatform.Probe()` journalise un résumé du type :

```text
Linux/linux-x64: GLFW=ok, Vulkan=ok, SIMD=avx2, primary=GLFW
```

Si Vulkan ou GLFW manque, Studio reste utilisable pour l’édition (scène, lois, blueprints, LLM) ; le viewport 3D et le mode `--engine` nécessitent un loader Vulkan résolvable.

## QA multi-RID (smoke)

```bash
# Publish les 6 RID et vérifie la présence de l’entrypoint (sans lancer Vulkan)
bash scripts/smoke-publish-rids.sh
```

CI : job `publish-smoke` dans `.github/workflows/build.yml`.

Checklist release et surfaces expérimentales : **[PRODUCTION.md](PRODUCTION.md)**.

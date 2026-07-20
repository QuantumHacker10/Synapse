# Captures d'écran — Synapse Studio v2.2

Visuels **PNG** représentatifs de l'interface Synapse Studio.

## Captures disponibles

| Fichier | Source |
|---|---|
| `studio-main-view.png` | Script Python (`generate-studio-screenshots.py`) |
| `studio-rendering.png` | Script Python |
| `studio-live.png` | Capture live Avalonia headless (`--screenshot`) |

## Capture live (v2.2)

```bash
dotnet run --project src/Synapse.Studio -- --screenshot docs/screenshots/studio-live.png --headless
```

Utilise **Avalonia Headless + Skia** pour produire un PNG fidèle à l'interface Studio (sans fenêtre visible).

## Régénérer les maquettes Python

```bash
python3 scripts/generate-studio-screenshots.py
```

## Éléments visibles

| Zone | Description |
|---|---|
| Panneau gauche | Hiérarchie des entités (Mesh, Agent, Genome) |
| Viewport central | Rendu Vulkan embarqué avec grille et gizmos |
| Panneau droit | Inspecteur (propriétés, loi physique active) |
| Barre inférieure | IPS, charge physique, preset L-DNN, console LLM |

Les SVG originaux (`studio-main-view.svg`, `studio-rendering.svg`) restent disponibles comme source vectorielle.

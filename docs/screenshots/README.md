# Captures d'écran — Synapse Studio

Visuels représentatifs de l'interface Synapse Studio v1.3.

> Les SVG ci-dessous sont des maquettes vectorielles fidèles à la disposition réelle de Studio.
> Pour des captures PNG depuis votre installation, lancez Studio et utilisez votre outil de capture d'écran.

## Vue principale

Interface complète : hiérarchie de scène, viewport Vulkan 3D, inspecteur, performance et console LLM.

![Vue principale de Synapse Studio](studio-main-view.svg)

## Rendu G-DNN + L-DNN

Viewport en mode rendu : forme SDF neuronale (G-DNN), illumination globale (L-DNN), SSAO et brouillard.

![Rendu neural temps réel](studio-rendering.svg)

## Capturer vos propres screenshots

```bash
# Lancer Studio avec la scène d'exemple
dotnet run --project src/Synapse.Studio -- --scene samples/demo.synapse

# Windows : Win+Shift+S
# Linux   : gnome-screenshot ou Flameshot
# macOS   : Cmd+Shift+4
```

Placez les PNG dans ce dossier et mettez à jour ce README.

## Éléments visibles

| Zone | Description |
|---|---|
| Panneau gauche | Hiérarchie des entités (Mesh, Character, Genome) |
| Viewport central | Rendu Vulkan embarqué avec grille et gizmos |
| Panneau droit | Inspecteur (propriétés, loi physique active) |
| Barre inférieure | IPS, charge physique, preset L-DNN, console LLM |

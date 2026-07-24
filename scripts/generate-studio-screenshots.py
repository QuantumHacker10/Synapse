#!/usr/bin/env python3
"""Generate PNG screenshots for Synapse Studio README (v2.1)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_main_view(path: Path) -> None:
    img = Image.new("RGB", (1280, 720), "#0E1218")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title = load_font(18)

    # Menu bar
    draw.rectangle((0, 0, 1280, 28), fill="#161B22")
    draw.text((12, 6), "Fichier   Simulation   Aide", fill="#C9D1D9", font=font)

    # Left panel
    draw.rectangle((0, 28, 300, 680), fill="#12171F", outline="#1E2633")
    draw.text((16, 40), "Hiérarchie", fill="#45E0B8", font=title)
    for i, name in enumerate(["Ground (Mesh)", "Agent_Alpha", "Agent_Beta", "NeuralForm (Genome)"]):
        draw.text((24, 78 + i * 26), name, fill="#8B9BB4", font=font)

    # Right inspector
    draw.rectangle((940, 28, 1280, 680), fill="#12171F", outline="#1E2633")
    draw.text((956, 40), "Inspecteur", fill="#45E0B8", font=title)
    draw.text((956, 78), "Loi active: heat_equation", fill="#C9D1D9", font=font)
    draw.text((956, 104), "Temp. moyenne: 300 K", fill="#8B9BB4", font=font)
    draw.text((956, 130), "Entités: 4", fill="#8B9BB4", font=font)

    # Viewport
    draw.rectangle((300, 28, 940, 680), fill="#0A0E14", outline="#1E2633")
    for x in range(320, 920, 40):
        draw.line((x, 40, x, 660), fill="#151B24", width=1)
    for y in range(40, 660, 40):
        draw.line((320, y, 920, y), fill="#151B24", width=1)
    draw.ellipse((560, 260, 680, 420), outline="#45E0B8", width=2)
    draw.text((430, 300), "Viewport Vulkan 3D", fill="#6E7681", font=title)

    # Status bar
    draw.rectangle((0, 680, 1280, 720), fill="#161B22")
    draw.text((12, 692), "60 FPS", fill="#8B9BB4", font=font)
    draw.text((140, 692), "Physique 2.1 ms", fill="#8B9BB4", font=font)
    draw.text((320, 692), "Simulation 1.8 ms", fill="#8B9BB4", font=font)
    draw.text((520, 692), "Qualité: High", fill="#8B9BB4", font=font)
    draw.text((700, 692), "LLM: Ollama local", fill="#8B9BB4", font=font)
    draw.text((1050, 692), "heat_equation", fill="#45E0B8", font=font)

    draw.text((12, 4), "Synapse Studio — SYNAPSE OMNIA v2.0", fill="#6E7681", font=load_font(11))
    img.save(path, format="PNG", optimize=True)


def draw_rendering_view(path: Path) -> None:
    img = Image.new("RGB", (1280, 720), "#05070A")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    title = load_font(20)

    # Gradient-like background bands
    for i, color in enumerate(["#0B1020", "#101830", "#0A1428", "#060A14"]):
        draw.rectangle((0, i * 180, 1280, (i + 1) * 180), fill=color)

    draw.ellipse((420, 180, 860, 560), fill="#1A2744", outline="#6FA8FF", width=3)
    draw.ellipse((500, 260, 780, 480), fill="#243B66", outline="#45E0B8", width=2)
    draw.text((470, 40), "G-DNN + L-DNN — Rendu neural temps réel", fill="#45E0B8", font=title)
    draw.text((470, 620), "SDF neuronale · GI hybride · SSAO · brouillard volumétrique", fill="#8B9BB4", font=font)

    # Fake light shafts
    for x in range(480, 820, 30):
        draw.line((x, 120, x + 40, 680), fill="#1E3A5F", width=8)

    img.save(path, format="PNG", optimize=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    draw_main_view(OUT / "studio-main-view.png")
    draw_rendering_view(OUT / "studio-rendering.png")
    print(f"Generated PNG screenshots in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

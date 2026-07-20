#!/usr/bin/env python3
"""Split monolithic C# source files into multiple files (same namespace, zero behavior change)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TYPE_PATTERN = re.compile(
    r"^\s*(?:\[.*\]\s*)*(?:public|internal|private|protected)?\s*"
    r"(?:readonly\s+|sealed\s+|static\s+|partial\s+|abstract\s+)*"
    r"(?:record\s+struct|record\s+class|record|class|struct|enum|interface)\s+(\w+)"
)
SECTION_PATTERN = re.compile(r"^//\s*=+")


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", name)
    return cleaned or "Section"


def split_by_regions(source: Path, output_prefix: str, output_dir: Path) -> int:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()

    namespace_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("namespace "))
    header = lines[: namespace_idx + 1]
    footer = ["}"]

    body_start = namespace_idx + 1
    if body_start < len(lines) and lines[body_start].strip() == "{":
        body_start += 1
    body = lines[body_start : -1]
    regions: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    for line in body:
        region_match = re.match(r"\s*#region\s+(.+)", line)
        if region_match:
            if current_name is not None:
                regions.append((current_name, current_lines))
            current_name = region_match.group(1).strip()
            current_lines = []
            continue
        if re.match(r"\s*#endregion", line):
            if current_name is not None:
                regions.append((current_name, current_lines))
                current_name = None
                current_lines = []
            continue
        if current_name is not None:
            current_lines.append(line)

    if not regions:
        raise RuntimeError(f"No #region blocks found in {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for region_name, region_lines in regions:
        file_name = f"{output_prefix}.{sanitize_name(region_name)}.cs"
        out_path = output_dir / file_name
        chunk = header + ["{"] + region_lines + footer
        out_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        written += 1

    source.unlink()
    return written


def _split_body_into_types(body: list[str]) -> list[tuple[str, list[str]]]:
    chunks: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    pending_prefix: list[str] = []
    depth = 0
    paren_depth = 0

    def flush() -> None:
        nonlocal current_name, current_lines
        if current_name and current_lines:
            chunks.append((current_name, current_lines))
        current_name = None
        current_lines = []

    for line in body:
        if SECTION_PATTERN.match(line) and depth == 0 and current_name is None:
            continue

        match = TYPE_PATTERN.match(line)
        if match and depth == 0 and current_name is None and paren_depth == 0:
            current_name = match.group(1)
            current_lines = pending_prefix + [line]
            pending_prefix = []
            depth = line.count("{") - line.count("}")
            paren_depth = line.count("(") - line.count(")")
            continue

        if current_name is not None:
            current_lines.append(line)
            depth += line.count("{") - line.count("}")
            paren_depth += line.count("(") - line.count(")")
            stripped = line.strip()
            if depth == 0 and paren_depth == 0 and (
                stripped == "}" or stripped.endswith(";")
            ):
                flush()
        else:
            pending_prefix.append(line)

    flush()
    return chunks


def split_by_top_level_types(source: Path, output_prefix: str, output_dir: Path) -> int:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()

    namespace_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("namespace "))
    header = lines[: namespace_idx + 1]
    footer = ["}"]

    body_start = namespace_idx + 1
    if body_start < len(lines) and lines[body_start].strip() == "{":
        body_start += 1
    body = lines[body_start : -1]
    chunks = _split_body_into_types(body)

    if not chunks:
        raise RuntimeError(f"No top-level types found in {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    written = 0
    for type_name, type_lines in chunks:
        count = seen.get(type_name, 0)
        seen[type_name] = count + 1
        suffix = "" if count == 0 else f"_{count + 1}"
        file_name = f"{output_prefix}.{type_name}{suffix}.cs"
        out_path = output_dir / file_name
        chunk = header + ["{"] + type_lines + footer
        out_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        written += 1

    source.unlink()
    return written


def split_file_scoped_by_sections(source: Path, output_prefix: str, output_dir: Path) -> int:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()

    namespace_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("namespace "))
    if not lines[namespace_idx].strip().endswith(";"):
        raise RuntimeError(f"{source} is not file-scoped namespace")

    header = lines[: namespace_idx + 1]
    body = lines[namespace_idx + 1 :]

    section_pattern = re.compile(r"^//\s*SECTION\s+(\d+)")
    sections: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []

    for line in body:
        match = section_pattern.match(line.strip())
        if match:
            if current_name is not None:
                sections.append((current_name, current_lines))
            current_name = f"Section{match.group(1)}"
            current_lines = [line]
            continue
        if current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        sections.append((current_name, current_lines))

    if not sections:
        raise RuntimeError(f"No SECTION markers found in {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for section_name, section_lines in sections:
        file_name = f"{output_prefix}.{section_name}.cs"
        out_path = output_dir / file_name
        chunk = header + [""] + section_lines
        out_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        written += 1

    source.unlink()
    return written


def split_file_scoped_by_top_level_types(source: Path, output_prefix: str, output_dir: Path) -> int:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()

    namespace_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("namespace "))
    if not lines[namespace_idx].strip().endswith(";"):
        raise RuntimeError(f"{source} is not file-scoped namespace")

    header = lines[: namespace_idx + 1]
    body = lines[namespace_idx + 1 :]
    chunks = _split_body_into_types(body)

    if not chunks:
        raise RuntimeError(f"No top-level types found in {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen: dict[str, int] = {}
    written = 0
    for type_name, type_lines in chunks:
        count = seen.get(type_name, 0)
        seen[type_name] = count + 1
        suffix = "" if count == 0 else f"_{count + 1}"
        file_name = f"{output_prefix}.{type_name}{suffix}.cs"
        out_path = output_dir / file_name
        chunk = header + [""] + type_lines
        out_path.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        written += 1

    source.unlink()
    return written


def main() -> int:
    jobs = [
        (
            ROOT / "src/Synapse.Physics/LivingLawLibrary.cs",
            "LivingLawLibrary",
            ROOT / "src/Synapse.Physics",
            split_file_scoped_by_sections,
        ),
        (
            ROOT / "src/Synapse.Simulation/EntityBehaviorSystem.cs",
            "EntityBehaviorSystem",
            ROOT / "src/Synapse.Simulation",
            split_by_regions,
        ),
    ]

    for source, prefix, out_dir, splitter in jobs:
        if not source.exists():
            print(f"SKIP (missing): {source}")
            continue
        count = splitter(source, prefix, out_dir)
        print(f"Split {source.name} -> {count} files in {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

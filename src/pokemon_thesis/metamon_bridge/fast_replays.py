"""
Fast discovery of parsed replay files without building the full Metamon index.

The ParsedReplayDataset scans every file on first load, which can take many
minutes on Windows. For early research, glob a small subset directly from disk.
"""

from __future__ import annotations

import json
import lz4.frame
from pathlib import Path


def discover_replay_files(
    cache_dir: Path,
    format_name: str,
    max_files: int,
    seed: int = 0,
) -> list[Path]:
    root = cache_dir / "parsed-replays" / format_name
    if not root.is_dir():
        raise FileNotFoundError(f"No replay folder at {root}")

    picked: list[Path] = []
    seen_years = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if not seen_years:
        raise FileNotFoundError(f"No year folders under {root}")

    year_index = seed % len(seen_years)
    for offset in range(len(seen_years)):
        year_dir = seen_years[(year_index + offset) % len(seen_years)]
        month_dirs = sorted(p for p in year_dir.iterdir() if p.is_dir())
        for month_dir in month_dirs:
            for path in sorted(month_dir.glob("*.json.lz4")):
                picked.append(path)
                if len(picked) >= max_files:
                    return picked

    if not picked:
        raise FileNotFoundError(f"No .json.lz4 files under {root}")
    return picked


def load_replay_json(path: Path) -> dict:
    with lz4.frame.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))

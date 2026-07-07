"""
Configure Metamon cache and optionally download human replay data.

Uses the Metamon install in pokemon-thesis-tools/metamon.
Run with the metamon venv OR after pip install -e pokemon-thesis-tools/metamon.

Examples
  python scripts/setup_metamon.py --check
  python scripts/setup_metamon.py --download gen9ou
  python scripts/setup_metamon.py --download gen9ou --fresh
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data" / "metamon"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up Metamon cache and datasets.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="METAMON_CACHE_DIR (default data/metamon, needs tens of GB for full sets)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify import and print dataset versions only.",
    )
    parser.add_argument(
        "--download",
        nargs="*",
        metavar="FORMAT",
        help="Download parsed human replays for formats (e.g. gen9ou).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing format folders first (fixes failed partial downloads on Windows).",
    )
    return parser.parse_args()


def env_with_cache(cache_dir: Path) -> dict[str, str]:
    merged = os.environ.copy()
    merged["METAMON_CACHE_DIR"] = str(cache_dir.resolve())
    return merged


def _win_long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if resolved.startswith("\\\\?\\"):
        return resolved
    return "\\\\?\\" + resolved


def _delete_tree_long(path: Path) -> bool:
    """Delete bottom-up using extended-length paths (Windows MAX_PATH workaround)."""
    root = _win_long_path(path)
    if not os.path.isdir(root):
        return not path.exists()

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            try:
                os.chmod(file_path, stat.S_IWRITE)
            except OSError:
                pass
            try:
                os.remove(file_path)
            except OSError:
                return False
        for name in dirnames:
            dir_path = os.path.join(dirpath, name)
            try:
                os.rmdir(dir_path)
            except OSError:
                return False
    try:
        os.rmdir(root)
    except OSError:
        return False
    return not path.exists()


def _robocopy_clear(path: Path) -> bool:
    """Mirror an empty folder onto target (common Windows bulk-delete workaround)."""
    empty = Path(tempfile.mkdtemp(prefix="metamon_empty_"))
    try:
        subprocess.run(
            [
                "robocopy",
                str(empty),
                str(path),
                "/MIR",
                "/R:1",
                "/W:1",
                "/NFL",
                "/NDL",
                "/NJH",
                "/NJS",
                "/NC",
                "/NS",
            ],
            check=False,
        )
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(path)], check=False)
        return not path.exists()
    finally:
        shutil.rmtree(empty, ignore_errors=True)


def _rename_as_stale(path: Path) -> Path:
    stale = path.parent / f"{path.name}.stale.{int(time.time())}"
    print(f"Rename leftover folder to {stale}")
    path.rename(stale)
    return stale


def remove_tree_windows_safe(path: Path) -> None:
    """Remove or sideline a large directory tree on Windows."""
    if not path.exists():
        return

    print(f"Removing {path} (this may take several minutes) ...")

    if sys.platform == "win32":
        if _delete_tree_long(path):
            return
        print("Long-path delete incomplete, trying robocopy ...")
        if _robocopy_clear(path):
            return
        if not path.exists():
            return
        try:
            _rename_as_stale(path)
            return
        except OSError as exc:
            raise RuntimeError(
                f"Could not delete or rename {path}. Another process may be "
                f"locking files. Close other terminals, then either delete the "
                f"folder in File Explorer or use a new cache directory:\n"
                f"  python scripts/setup_metamon.py --download gen9ou "
                f"--cache-dir data/metamon_clean"
            ) from exc

    def onerror(func, target, exc_info):
        if not os.path.exists(target):
            return
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=onerror)


def download_formats(formats: list[str], cache_dir: Path, fresh: bool) -> None:
    os.environ["METAMON_CACHE_DIR"] = str(cache_dir.resolve())

    from metamon.data.download import download_parsed_replays

    parsed_root = cache_dir / "parsed-replays"
    for battle_format in formats:
        out_path = parsed_root / battle_format
        tar_path = parsed_root / f"{battle_format}.tar.gz"

        if fresh:
            print(f"Removing {out_path} ...")
            remove_tree_windows_safe(out_path)
            if tar_path.exists():
                tar_path.unlink()

        force = fresh
        if out_path.exists() and not fresh:
            print(f"Resuming or skipping existing data at {out_path}")
            force = False

        print(f"Downloading parsed replays for {battle_format} (force={force}) ...")
        download_parsed_replays(battle_format, force_download=force)
        print(f"Ready at {out_path}")


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = env_with_cache(cache_dir)

    print(f"METAMON_CACHE_DIR  {cache_dir}")

    if args.check or not args.download:
        subprocess.run(
            [sys.executable, "-m", "metamon.data.download", "check-versions"],
            check=True,
            cwd=ROOT,
            env=env,
        )

    if args.check and not args.download:
        _probe_dataset()
        return

    if args.download is not None:
        formats = args.download if args.download else ["gen9ou"]
        print("This can take a long time and use many gigabytes of disk.")
        if not args.fresh:
            print("Using resume mode. Pass --fresh to wipe and re-download.")
        download_formats(formats, cache_dir, fresh=args.fresh)
        print("Download finished for", ", ".join(formats))


def _probe_dataset() -> None:
    try:
        from metamon.data import ParsedReplayDataset
        from metamon.interface import (
            DefaultActionSpace,
            DefaultObservationSpace,
            DefaultShapedReward,
        )
    except ImportError as exc:
        print("Metamon import failed.", exc)
        print("Install with  pip install -e pokemon-thesis-tools/metamon")
        raise SystemExit(1) from exc

    formats = ["gen9ou"]
    print("Probing ParsedReplayDataset for", formats)
    dset = ParsedReplayDataset(
        observation_space=DefaultObservationSpace(),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=formats,
    )
    obs, actions, rewards, dones = dset[0]
    print(f"Sample battle length  {len(obs)} turns")
    print(f"Observation dim       {obs[0].shape if hasattr(obs[0], 'shape') else type(obs[0])}")
    print(f"Action space size     {len(actions)}")


if __name__ == "__main__":
    main()

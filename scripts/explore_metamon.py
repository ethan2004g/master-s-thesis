"""
Inspect one Metamon human replay trajectory.

Requires parsed replays for the chosen format (see setup_metamon.py).

Example
  set METAMON_CACHE_DIR=C:\\Users\\ethan\\Masters Thesis\\data\\metamon
  python scripts/explore_metamon.py --format gen9ou
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CACHE = ROOT / "data" / "metamon"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore one Metamon replay trajectory.")
    parser.add_argument("--format", default="gen9ou", help="Showdown format (default gen9ou)")
    parser.add_argument("--index", type=int, default=0, help="Dataset index (default 0)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="METAMON_CACHE_DIR if not already set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir.resolve())
    elif "METAMON_CACHE_DIR" not in os.environ:
        os.environ["METAMON_CACHE_DIR"] = str(DEFAULT_CACHE.resolve())

    from metamon.data import ParsedReplayDataset
    from metamon.interface import (
        DefaultActionSpace,
        DefaultObservationSpace,
        DefaultShapedReward,
    )

    dset = ParsedReplayDataset(
        observation_space=DefaultObservationSpace(),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=[args.format],
    )

    print(f"Dataset size          {len(dset)} battles")
    print(f"METAMON_CACHE_DIR     {os.environ['METAMON_CACHE_DIR']}")

    obs_seq, action_seq, reward_seq, done_seq = dset[args.index]
    n_turns = len(reward_seq)
    print(f"Battle index          {args.index}")
    print(f"Turns                 {n_turns}")
    print(f"Total reward          {float(reward_seq.sum()):.3f}")
    print(f"Done flags (last 5)   {list(done_seq[-5:])}")
    print(f"Chosen actions        {len(action_seq.get('chosen', []))}")
    print(f"Missing actions       {sum(action_seq.get('missing', []))}")

    if obs_seq:
        sample_key = next(iter(obs_seq))
        first_step = obs_seq[sample_key][0]
        print(f"Observation keys      {len(obs_seq)}")
        print(f"Sample key            {sample_key}")
        print(f"First step type       {type(first_step).__name__}")
        if hasattr(first_step, "shape"):
            print(f"First step shape      {first_step.shape}")


if __name__ == "__main__":
    main()

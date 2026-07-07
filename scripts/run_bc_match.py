"""
Run a trained BCPlayer against a RandomPlayer on the local Showdown server.

Example
  python scripts/run_bc_match.py --battles 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poke_env.player import RandomPlayer

from pokemon_thesis.inference import BCPlayer


async def run_match(n_battles: int, checkpoint_dir: Path, battle_format: str) -> None:
    bc_player = BCPlayer(
        checkpoint_dir=checkpoint_dir,
        max_concurrent_battles=1,
        battle_format=battle_format,
    )
    opponent = RandomPlayer(
        max_concurrent_battles=1,
        battle_format=battle_format,
    )

    try:
        await bc_player.battle_against(opponent, n_battles=n_battles)
    finally:
        await bc_player.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()

    print(f"Battles finished {bc_player.n_finished_battles}")
    print(f"BC wins          {bc_player.n_won_battles}")
    print(f"BC win rate      {bc_player.win_rate:.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BCPlayer vs RandomPlayer matches.")
    parser.add_argument("--battles", type=int, default=5)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/human_bc"),
    )
    parser.add_argument(
        "--battle-format",
        default="gen9randombattle",
        help="Showdown format for live matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(run_match(args.battles, args.checkpoint_dir, args.battle_format))


if __name__ == "__main__":
    main()

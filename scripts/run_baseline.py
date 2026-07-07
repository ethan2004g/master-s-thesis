"""
Phase 1 baseline — RandomPlayer vs RandomPlayer with per-turn JSONL logging.

Requires a local Showdown server:
  node pokemon-showdown start --no-security
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from poke_env.battle import AbstractBattle
from poke_env.player import RandomPlayer

from pokemon_thesis.battle_snapshot import battle_snapshot, describe_order


class TurnLogWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._handle = path.open("w", encoding="utf-8")
        self.turns_logged = 0

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.turns_logged += 1

    def close(self) -> None:
        self._handle.close()

    @property
    def path(self) -> Path:
        return self._path


class LoggingRandomPlayer(RandomPlayer):
    def __init__(
        self,
        turn_log: TurnLogWriter,
        stop_after_turns: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._turn_log = turn_log
        self._stop_after_turns = stop_after_turns

    def choose_move(self, battle: AbstractBattle):
        order = super().choose_move(battle)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "player": self.username,
            "action": describe_order(order),
            **battle_snapshot(battle),
        }
        self._turn_log.write(record)
        return order

    @property
    def should_stop_battles(self) -> bool:
        if self._stop_after_turns is None:
            return False
        return self._turn_log.turns_logged >= self._stop_after_turns


async def run_baseline(
    n_battles: int,
    min_turns: Optional[int],
    log_dir: Path,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"baseline_{stamp}.jsonl"
    turn_log = TurnLogWriter(log_path)

    player_a = LoggingRandomPlayer(
        turn_log=turn_log,
        stop_after_turns=min_turns,
        max_concurrent_battles=1,
        battle_format="gen9randombattle",
    )
    player_b = LoggingRandomPlayer(
        turn_log=turn_log,
        stop_after_turns=min_turns,
        max_concurrent_battles=1,
        battle_format="gen9randombattle",
    )

    battles_run = 0
    try:
        for _ in range(n_battles):
            if player_a.should_stop_battles:
                break
            await player_a.battle_against(player_b, n_battles=1)
            battles_run += 1
            if player_a.should_stop_battles:
                break
    finally:
        await player_a.ps_client.stop_listening()
        await player_b.ps_client.stop_listening()
        turn_log.close()

    print(f"Log file          {turn_log.path}")
    print(f"Battles completed {battles_run}")
    print(f"Turn decisions    {turn_log.turns_logged}")
    print(f"Player A wins     {player_a.n_won_battles}")
    print(f"Player B wins     {player_b.n_won_battles}")
    print(f"Player A win rate {player_a.win_rate:.1%}")
    return turn_log.path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RandomPlayer baseline battles with per-turn logging."
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=10,
        help="Maximum number of battles to run (default 10).",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=None,
        help="Stop after this many logged turn decisions across all battles.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for JSONL output (default logs/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        run_baseline(
            n_battles=args.battles,
            min_turns=args.min_turns,
            log_dir=args.log_dir,
        )
    )


if __name__ == "__main__":
    main()

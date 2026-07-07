"""
Export a subset of Metamon human replays to thesis JSONL (and optional encoded form).

Requires METAMON_CACHE_DIR with parsed gen9ou data.

Example
  $env:METAMON_CACHE_DIR = "C:\\Users\\ethan\\Masters Thesis\\data\\metamon_clean"
  .\\.venv-metamon\\Scripts\\Activate.ps1
  python scripts/export_metamon_subset.py --max-battles 10000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CACHE = ROOT / "data" / "metamon_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Metamon parsed replays to thesis training JSONL."
    )
    parser.add_argument(
        "--format",
        default="gen9ou",
        help="Showdown format (default gen9ou).",
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        default=10_000,
        help="How many battles to export (default 10000).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="METAMON_CACHE_DIR (default data/metamon_clean).",
    )
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=Path("logs/metamon_baseline.jsonl"),
        help="Baseline-shaped JSONL output.",
    )
    parser.add_argument(
        "--encoded-out",
        type=Path,
        default=Path("logs/encoded_metamon_subset.jsonl"),
        help="Encoded 13-token JSONL output.",
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("logs/vocab.json"),
        help="Token vocabulary path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for battle sampling.",
    )
    parser.add_argument(
        "--skip-encoded",
        action="store_true",
        help="Only write baseline JSONL, not encoded output.",
    )
    return parser.parse_args()


def battle_tag_from_path(path: str) -> str:
    name = Path(path).name
    if name.endswith(".json.lz4"):
        return name[: -len(".json.lz4")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return name


def main() -> None:
    args = parse_args()
    cache_dir = (args.cache_dir or DEFAULT_CACHE).resolve()
    os.environ["METAMON_CACHE_DIR"] = str(cache_dir)

    from metamon.data.parsed_replay_dset import ParsedReplayDataset
    from metamon.interface import (
        DefaultActionSpace,
        DefaultObservationSpace,
        DefaultShapedReward,
        UniversalState,
    )
    from pokemon_thesis.metamon_bridge import universal_state_to_record
    from pokemon_thesis.metamon_bridge.opponent_truth import attach_belief_and_matchup
    from pokemon_thesis.tokenizer import TurnTokenizer

    print(f"METAMON_CACHE_DIR  {cache_dir}")
    print(f"Format             {args.format}")
    print(f"Max battles        {args.max_battles}")

    dset = ParsedReplayDataset(
        observation_space=DefaultObservationSpace(),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=[args.format],
        verbose=True,
    )

    filenames = list(dset.filenames)
    rng = random.Random(args.seed)
    rng.shuffle(filenames)
    selected = filenames[: min(args.max_battles, len(filenames))]
    print(f"Total indexed      {len(filenames)}")
    print(f"Exporting          {len(selected)} battles")

    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_encoded:
        args.encoded_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = TurnTokenizer()
    battles_written = 0
    turns_written = 0
    turns_skipped = 0

    with args.baseline_out.open("w", encoding="utf-8") as baseline_sink:
        encoded_sink = None
        if not args.skip_encoded:
            encoded_sink = args.encoded_out.open("w", encoding="utf-8")

        try:
            for battle_i, filename in enumerate(selected):
                data = dset._load_json(filename)
                states = [UniversalState.from_dict(s) for s in data["states"]]
                action_indices = data["actions"][:-1]
                tag = battle_tag_from_path(filename)

                for turn_idx, action_idx in enumerate(action_indices):
                    if action_idx < 0:
                        turns_skipped += 1
                        continue

                    state = states[turn_idx]
                    record = universal_state_to_record(
                        state=state,
                        turn=turn_idx + 1,
                        battle_tag=tag,
                        action_idx=int(action_idx),
                    )
                    record = attach_belief_and_matchup(record, state, states)
                    baseline_sink.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
                    turns_written += 1

                    if encoded_sink is not None:
                        turn_tokens = tokenizer.encode_record(record)
                        encoded = {
                            "battle_tag": tag,
                            "turn": turn_idx + 1,
                            "player": record.get("player", "p1"),
                            "action": record["action"],
                            "token_strings": list(turn_tokens.token_strings),
                            "token_ids": list(turn_tokens.token_ids),
                            "belief": record.get("belief"),
                            "team_matchup": record.get("team_matchup"),
                        }
                        encoded_sink.write(
                            json.dumps(encoded, separators=(",", ":")) + "\n"
                        )

                battles_written += 1
                if (battle_i + 1) % 500 == 0:
                    print(
                        f"  progress {battle_i + 1}/{len(selected)} battles, "
                        f"{turns_written} turns"
                    )
        finally:
            if encoded_sink is not None:
                encoded_sink.close()

    tokenizer.vocab.save(args.vocab_out)

    print(f"Battles exported    {battles_written}")
    print(f"Turns exported      {turns_written}")
    print(f"Turns skipped       {turns_skipped} (hidden replay actions)")
    print(f"Baseline file       {args.baseline_out}")
    if not args.skip_encoded:
        print(f"Encoded file        {args.encoded_out}")
    print(f"Vocabulary          {args.vocab_out} ({len(tokenizer.vocab)} tokens)")


if __name__ == "__main__":
    main()

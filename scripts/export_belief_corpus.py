"""
Export human replays with active-opponent belief labels and team matchup keys.

Example
  .\\.venv-metamon\\Scripts\\Activate.ps1
  python scripts/export_belief_corpus.py --battles 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CACHE = ROOT / "data" / "metamon_clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export belief-enriched encoded corpus.")
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=Path("logs/belief_metamon_baseline.jsonl"),
    )
    parser.add_argument(
        "--encoded-out",
        type=Path,
        default=Path("logs/encoded_belief_corpus.jsonl"),
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("logs/vocab_belief.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir.resolve())

    from metamon.interface import UniversalState
    from pokemon_thesis.metamon_bridge.fast_replays import (
        discover_replay_files,
        load_replay_json,
    )
    from pokemon_thesis.metamon_bridge.opponent_truth import attach_belief_and_matchup
    from pokemon_thesis.metamon_bridge.state_to_record import universal_state_to_record
    from pokemon_thesis.tokenizer import TurnTokenizer

    files = discover_replay_files(args.cache_dir, args.format, args.battles, args.seed)
    print(f"Exporting {len(files)} battles from {args.cache_dir}")

    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    args.encoded_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = TurnTokenizer()
    turns_written = 0
    turns_skipped = 0
    belief_stats = {"move": 0, "item": 0, "ability": 0, "tera": 0}

    with args.baseline_out.open("w", encoding="utf-8") as baseline_sink, args.encoded_out.open(
        "w", encoding="utf-8"
    ) as encoded_sink:
        for file_path in files:
            data = load_replay_json(file_path)
            states = [UniversalState.from_dict(s) for s in data["states"]]
            action_indices = data["actions"][:-1]
            tag = file_path.stem

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
                baseline_sink.write(json.dumps(record, separators=(",", ":")) + "\n")

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
                encoded_sink.write(json.dumps(encoded, separators=(",", ":")) + "\n")
                turns_written += 1

                belief = record.get("belief") or {}
                belief_stats["move"] += sum(1 for flag in belief.get("move_mask") or [] if flag)
                belief_stats["item"] += int(bool(belief.get("item_mask")))
                belief_stats["ability"] += int(bool(belief.get("ability_mask")))
                belief_stats["tera"] += int(bool(belief.get("tera_mask")))

    tokenizer.vocab.save(args.vocab_out)

    print(f"Turns exported        {turns_written}")
    print(f"Turns skipped         {turns_skipped}")
    print(f"Supervised move slots {belief_stats['move']}")
    print(f"Supervised item turns {belief_stats['item']}")
    print(f"Supervised ability    {belief_stats['ability']}")
    print(f"Supervised tera       {belief_stats['tera']}")
    print(f"Baseline JSONL        {args.baseline_out}")
    print(f"Encoded JSONL         {args.encoded_out}")
    print(f"Vocabulary            {args.vocab_out}")


if __name__ == "__main__":
    main()

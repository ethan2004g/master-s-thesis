"""
Phase 1 research kickoff.

Exports a small human-replay subset from Metamon (fast path, no full index),
trains BC on human data, then trains BeliefTransformer when belief labels exist.

Example
  .\\.venv-metamon\\Scripts\\Activate.ps1
  python scripts/start_research.py --battles 200

Use the thesis venv for training only if Metamon is not installed there.
This script calls each venv's Python explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CACHE = ROOT / "data" / "metamon_clean"
METAMON_PY = ROOT / ".venv-metamon" / "Scripts" / "python.exe"
THESIS_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run phase 1 research pipeline.")
    parser.add_argument("--battles", type=int, default=200, help="Battles to export.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--format", default="gen9ou")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs-bc", type=int, default=15)
    parser.add_argument("--epochs-belief", type=int, default=15)
    parser.add_argument("--skip-train", action="store_true", help="Export only.")
    parser.add_argument(
        "--baseline-out",
        type=Path,
        default=Path("logs/research_metamon_baseline.jsonl"),
    )
    parser.add_argument(
        "--encoded-out",
        type=Path,
        default=Path("logs/encoded_research_metamon.jsonl"),
    )
    parser.add_argument(
        "--vocab-out",
        type=Path,
        default=Path("logs/vocab.json"),
    )
    return parser.parse_args()


def check_prerequisites(cache_dir: Path) -> None:
    if not METAMON_PY.is_file():
        raise SystemExit(f"Missing Metamon venv Python at {METAMON_PY}")
    if not THESIS_PY.is_file():
        raise SystemExit(f"Missing thesis venv Python at {THESIS_PY}")
    replay_root = cache_dir / "parsed-replays" / "gen9ou"
    if not replay_root.is_dir():
        raise SystemExit(
            f"No gen9ou replays at {replay_root}. "
            "Run setup_metamon.py --download gen9ou first."
        )


def export_subset(args: argparse.Namespace) -> dict[str, int]:
    os.environ["METAMON_CACHE_DIR"] = str(args.cache_dir.resolve())

    from metamon.interface import UniversalState
    from pokemon_thesis.metamon_bridge.opponent_truth import attach_belief_and_matchup
    from pokemon_thesis.metamon_bridge.fast_replays import (
        discover_replay_files,
        load_replay_json,
    )
    from pokemon_thesis.metamon_bridge.state_to_record import universal_state_to_record
    from pokemon_thesis.tokenizer import TurnTokenizer

    files = discover_replay_files(
        args.cache_dir, args.format, max_files=args.battles, seed=args.seed
    )
    print(f"Fast export from     {args.cache_dir}")
    print(f"Replay files picked  {len(files)}")

    args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
    args.encoded_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = TurnTokenizer()
    battles_written = 0
    turns_written = 0
    turns_skipped = 0
    belief_turns = 0

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
                if (record.get("belief") or {}).get("opponent_species_mask"):
                    if any(record["belief"]["opponent_species_mask"]):
                        belief_turns += 1

            battles_written += 1

    tokenizer.vocab.save(args.vocab_out)

    stats = {
        "battles": battles_written,
        "turns": turns_written,
        "turns_skipped": turns_skipped,
        "belief_turns": belief_turns,
    }
    print(f"Battles exported      {stats['battles']}")
    print(f"Turns exported        {stats['turns']}")
    print(f"Turns skipped         {stats['turns_skipped']}")
    print(f"Turns with belief sup {stats['belief_turns']}")
    print(f"Baseline JSONL        {args.baseline_out}")
    print(f"Encoded JSONL         {args.encoded_out}")
    print(f"Vocabulary            {args.vocab_out} ({len(tokenizer.vocab)} tokens)")
    return stats


def run_training(args: argparse.Namespace) -> None:
    encoded = str(args.encoded_out.resolve())
    vocab = str(args.vocab_out.resolve())

    print("\n--- Training BC TurnTransformer on human replays ---")
    subprocess.run(
        [
            str(THESIS_PY),
            str(ROOT / "scripts" / "train_bc.py"),
            encoded,
            "--token-vocab",
            vocab,
            "--epochs",
            str(args.epochs_bc),
            "--out-dir",
            str(ROOT / "checkpoints" / "research_bc"),
        ],
        cwd=ROOT,
        check=True,
    )

    print("\n--- Training BeliefTransformer ---")
    result = subprocess.run(
        [
            str(THESIS_PY),
            str(ROOT / "scripts" / "train_belief_bc.py"),
            encoded,
            "--token-vocab",
            vocab,
            "--epochs",
            str(args.epochs_belief),
            "--out-dir",
            str(ROOT / "checkpoints" / "research_belief"),
        ],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(
            "Belief training skipped or failed. "
            "This is expected if no supervised belief slots were exported."
        )


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    check_prerequisites(args.cache_dir)
    stats = export_subset(args)

    summary_path = ROOT / "logs" / "research_phase1_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "battles": stats["battles"],
                "turns": stats["turns"],
                "belief_turns": stats["belief_turns"],
                "encoded_out": str(args.encoded_out),
                "baseline_out": str(args.baseline_out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.skip_train:
        print(f"\nExport complete. Summary at {summary_path}")
        return

    if stats["turns"] == 0:
        raise SystemExit("No turns exported. Cannot train.")

    run_training(args)
    print(f"\nPhase 1 complete. Summary at {summary_path}")


if __name__ == "__main__":
    main()

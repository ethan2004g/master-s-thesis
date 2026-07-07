"""
Encode baseline JSONL logs into 13-token turn sequences.

Example
  python scripts/encode_logs.py logs/baseline_20260529T011709Z.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokemon_thesis.tokenizer import TurnTokenizer, TokenVocab


def _make_tokenizer(vocab_path: Path) -> TurnTokenizer:
    if vocab_path.exists():
        return TurnTokenizer(vocab=TokenVocab.load(vocab_path))
    return TurnTokenizer()


def encode_file(
    input_path: Path,
    output_path: Path,
    vocab_path: Path,
) -> None:
    tokenizer = _make_tokenizer(vocab_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            turn_tokens = tokenizer.encode_record(record)
            encoded = {
                "battle_tag": record.get("battle_tag"),
                "turn": record.get("turn"),
                "player": record.get("player"),
                "action": record.get("action"),
                "token_strings": list(turn_tokens.token_strings),
                "token_ids": list(turn_tokens.token_ids),
            }
            sink.write(json.dumps(encoded, separators=(",", ":")) + "\n")
            records_written += 1

    tokenizer.vocab.save(vocab_path)

    print(f"Input             {input_path}")
    print(f"Output            {output_path}")
    print(f"Vocabulary        {vocab_path}")
    print(f"Records encoded   {records_written}")
    print(f"Vocabulary size   {len(tokenizer.vocab)}")


def encode_corpus(
    input_paths: list[Path],
    output_path: Path,
    vocab_path: Path,
) -> int:
    """Encode many baseline logs into one file with a shared vocabulary."""
    if not input_paths:
        return 0

    tokenizer = _make_tokenizer(vocab_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_written = 0

    with output_path.open("w", encoding="utf-8") as sink:
        for input_path in input_paths:
            with input_path.open(encoding="utf-8") as source:
                for line in source:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    turn_tokens = tokenizer.encode_record(record)
                    encoded = {
                        "battle_tag": record.get("battle_tag"),
                        "turn": record.get("turn"),
                        "player": record.get("player"),
                        "action": record.get("action"),
                        "token_strings": list(turn_tokens.token_strings),
                        "token_ids": list(turn_tokens.token_ids),
                    }
                    sink.write(json.dumps(encoded, separators=(",", ":")) + "\n")
                    records_written += 1

    tokenizer.vocab.save(vocab_path)
    return records_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode baseline JSONL logs into 13-token sequences."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a baseline JSONL file from run_baseline.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (default logs/encoded_<input stem>.jsonl)",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("logs/vocab.json"),
        help="Where to write the token vocabulary (default logs/vocab.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.parent / f"encoded_{args.input.stem}.jsonl"
    encode_file(args.input, output, args.vocab)


if __name__ == "__main__":
    main()

"""
Run battle collection, encoding, and BC training in a loop until you press Ctrl+C.

Requires the local Showdown server to be running.

Example
  python scripts/continuous_train.py --battles-per-cycle 10 --epochs 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PYTHON = sys.executable
sys.path.insert(0, str(SRC))


def _subprocess_env() -> dict[str, str]:
    import os

    env = os.environ.copy()
    src = str(SRC)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env


def encode_corpus(input_paths: list[Path], output_path: Path, vocab_path: Path) -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "encode_logs", ROOT / "scripts" / "encode_logs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.encode_corpus(input_paths, output_path, vocab_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously collect battles, encode, and train BC until stopped."
    )
    parser.add_argument(
        "--battles-per-cycle",
        type=int,
        default=10,
        help="Random vs random battles per cycle (default 10).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs per cycle (default 15).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8,
        help="Context window size (default 8).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for logs and encoded corpus.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to pause between cycles (default 0).",
    )
    parser.add_argument(
        "--eval-battles",
        type=int,
        default=0,
        help="Optional BC vs Random battles after each train cycle.",
    )
    return parser.parse_args()


def run_baseline(battles: int, log_dir: Path) -> None:
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--battles",
        str(battles),
        "--log-dir",
        str(log_dir),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=_subprocess_env())


def run_training(
    encoded_path: Path,
    vocab_path: Path,
    epochs: int,
    context: int,
) -> None:
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train_bc.py"),
        str(encoded_path),
        "--epochs",
        str(epochs),
        "--context",
        str(context),
        "--token-vocab",
        str(vocab_path),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=_subprocess_env())


def run_bc_eval(battles: int) -> None:
    cmd = [
        PYTHON,
        str(ROOT / "scripts" / "run_bc_match.py"),
        "--battles",
        str(battles),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, env=_subprocess_env())


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = log_dir / "encoded_corpus.jsonl"
    vocab_path = log_dir / "vocab.json"
    cycle = 0

    print("Continuous training started. Press Ctrl+C to stop.")
    print(f"Log directory     {log_dir}")
    print(f"Battles per cycle {args.battles_per_cycle}")
    print(f"Epochs per cycle  {args.epochs}")

    try:
        while True:
            cycle += 1
            started = time.strftime("%H:%M:%S")
            print()
            print(f"=== Cycle {cycle} started at {started} ===")

            run_baseline(args.battles_per_cycle, log_dir)

            baseline_files = sorted(log_dir.glob("baseline_*.jsonl"))
            record_count = encode_corpus(baseline_files, corpus_path, vocab_path)
            print(f"Corpus records    {record_count}")
            print(f"Corpus file       {corpus_path}")

            if record_count == 0:
                print("No records to train on yet. Starting next cycle.")
                continue

            run_training(corpus_path, vocab_path, args.epochs, args.context)

            if args.eval_battles > 0:
                print(f"Running {args.eval_battles} BC evaluation battles...")
                run_bc_eval(args.eval_battles)

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print()
        print(f"Stopped after {cycle} cycle(s). Latest checkpoint in checkpoints/")


if __name__ == "__main__":
    main()

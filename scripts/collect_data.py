"""
Collect baseline logs, encode them, and optionally retrain the BC model.

Example
  python scripts/collect_data.py --battles 50 --train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline battles, encode logs, and optionally train BC."
    )
    parser.add_argument("--battles", type=int, default=50)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--train", action="store_true", help="Train BC after encoding.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--context", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    baseline_cmd = [
        PYTHON,
        str(ROOT / "scripts" / "run_baseline.py"),
        "--battles",
        str(args.battles),
        "--log-dir",
        str(log_dir),
    ]
    print("Running", " ".join(baseline_cmd))
    subprocess.run(baseline_cmd, check=True, cwd=ROOT)

    baseline_files = sorted(log_dir.glob("baseline_*.jsonl"))
    if not baseline_files:
        raise SystemExit(f"No baseline logs found in {log_dir}")
    latest_log = baseline_files[-1]

    encode_cmd = [
        PYTHON,
        str(ROOT / "scripts" / "encode_logs.py"),
        str(latest_log),
        "--vocab",
        str(log_dir / "vocab.json"),
    ]
    print("Running", " ".join(encode_cmd))
    subprocess.run(encode_cmd, check=True, cwd=ROOT)

    encoded_path = log_dir / f"encoded_{latest_log.stem}.jsonl"
    print(f"Encoded file      {encoded_path}")

    if not args.train:
        print("Skipping training. Pass --train to fit BC on the new encoded log.")
        return

    train_cmd = [
        PYTHON,
        str(ROOT / "scripts" / "train_bc.py"),
        str(encoded_path),
        "--epochs",
        str(args.epochs),
        "--context",
        str(args.context),
        "--token-vocab",
        str(log_dir / "vocab.json"),
    ]
    print("Running", " ".join(train_cmd))
    subprocess.run(train_cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()

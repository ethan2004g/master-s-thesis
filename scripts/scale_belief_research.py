"""
Scale belief corpus and retrain all Phase 2 models.

Example
  .\\.venv-metamon\\Scripts\\Activate.ps1
  python scripts/scale_belief_research.py --battles 2000 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METAMON_PY = ROOT / ".venv-metamon" / "Scripts" / "python.exe"
THESIS_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export, train, and eval at scale.")
    parser.add_argument("--battles", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===")
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)

    corpus = ROOT / "logs" / "encoded_belief_corpus.jsonl"
    vocab = ROOT / "logs" / "vocab_belief.json"

    if not args.skip_export:
        env = os.environ.copy()
        env["METAMON_CACHE_DIR"] = str(ROOT / "data" / "metamon_clean")
        run(
            [
                str(METAMON_PY),
                str(ROOT / "scripts" / "export_belief_corpus.py"),
                "--battles",
                str(args.battles),
            ],
            label=f"Export {args.battles} battles",
        )

    if not args.skip_train:
        common = [
            str(corpus),
            "--token-vocab",
            str(vocab),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ]
        run(
            [
                str(THESIS_PY),
                str(ROOT / "scripts" / "train_bc.py"),
                *common,
                "--out-dir",
                str(ROOT / "checkpoints" / "human_bc_belief"),
            ],
            label="Train human_bc_belief",
        )
        run(
            [
                str(THESIS_PY),
                str(ROOT / "scripts" / "train_matchup_bc.py"),
                *common,
                "--out-dir",
                str(ROOT / "checkpoints" / "matchup_bc"),
            ],
            label="Train matchup_bc",
        )
        run(
            [
                str(THESIS_PY),
                str(ROOT / "scripts" / "train_active_belief.py"),
                *common,
                "--out-dir",
                str(ROOT / "checkpoints" / "active_belief"),
            ],
            label="Train active_belief",
        )

    if not args.skip_eval:
        run(
            [str(THESIS_PY), str(ROOT / "scripts" / "eval_all_ood.py")],
            label="OOD evaluation",
        )

    summary_path = ROOT / "logs" / "research_phase2_summary.json"
    if (ROOT / "logs" / "ood_comparison.json").is_file():
        comparison = json.loads(
            (ROOT / "logs" / "ood_comparison.json").read_text(encoding="utf-8")
        )
    else:
        comparison = {}
    summary = {
        "phase": 2,
        "status": "scaled" if not args.skip_train else "exported",
        "battles": args.battles,
        "epochs": args.epochs,
        "corpus": str(corpus),
        "ood_comparison": comparison,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary at {summary_path}")


if __name__ == "__main__":
    main()

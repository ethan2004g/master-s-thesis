"""
Run OOD team-split evaluation for all trained checkpoints on the belief corpus.

Example
  python scripts/eval_all_ood.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
CORPUS = ROOT / "logs" / "encoded_belief_corpus.jsonl"
VOCAB = ROOT / "logs" / "vocab_belief.json"

CHECKPOINTS = [
    ("human_bc_belief", "turn"),
    ("matchup_bc", "matchup"),
    ("active_belief", "active"),
]


def main() -> None:
    summary: dict[str, dict] = {}
    for name, _model in CHECKPOINTS:
        out = ROOT / "logs" / f"eval_{name}_ood.json"
        cmd = [
            str(PYTHON),
            str(ROOT / "scripts" / "eval_bc.py"),
            "--checkpoint-dir",
            str(ROOT / "checkpoints" / name),
            "--corpus",
            str(CORPUS),
            "--token-vocab",
            str(VOCAB),
            "--out",
            str(out),
        ]
        print(f"\n=== Evaluating {name} ===")
        subprocess.run(cmd, cwd=ROOT, check=True)
        summary[name] = json.loads(out.read_text(encoding="utf-8"))

    comparison = {
        "corpus": str(CORPUS),
        "models": {
            name: {
                "id_val_accuracy": data["id_val_split"]["accuracy"],
                "id_train_team_accuracy": data["id_train_team_split"]["accuracy"],
                "ood_team_accuracy": data["ood_team_split"]["accuracy"],
                "team_generalization_gap": (
                    data["id_train_team_split"]["accuracy"]
                    - data["ood_team_split"]["accuracy"]
                ),
                "ood_samples": data["ood_team_split"]["samples"],
                "held_out_team_keys": data.get("ood_team_keys_held_out"),
            }
            for name, data in summary.items()
        },
    }
    out_path = ROOT / "logs" / "ood_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\n=== Comparison ===")
    print(json.dumps(comparison, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

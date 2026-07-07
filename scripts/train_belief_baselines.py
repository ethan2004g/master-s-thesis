"""
Train MLP and GRU move-belief baselines on the belief corpus.

Example
  python scripts/train_belief_baselines.py logs/encoded_belief_corpus.jsonl --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokemon_thesis.data.action_vocab import ActionVocab
from pokemon_thesis.data.active_belief_dataset import ActiveBeliefDataset
from pokemon_thesis.data.active_belief_labels import build_active_belief_vocabs
from pokemon_thesis.data.encoded_dataset import load_encoded_records
from pokemon_thesis.model.belief_baselines import MoveBeliefGRU, MoveBeliefMLP
from pokemon_thesis.tokenizer.vocab import TokenVocab


def masked_move_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> tuple[float, int]:
    preds = logits.argmax(dim=-1)
    valid = mask.bool()
    if valid.sum().item() == 0:
        return 0.0, 0
    correct = (preds[valid] == labels[valid]).sum().item()
    count = valid.sum().item()
    return correct / count, count


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_correct = 0
    total_slots = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        move_labels = batch["move_labels"].to(device)
        move_mask = batch["move_mask"].to(device)

        if train_mode:
            optimizer.zero_grad()

        logits = model(token_ids, turn_mask)
        loss = 0.0
        for slot in range(4):
            slot_mask = move_mask[:, slot] > 0
            if slot_mask.any():
                loss = loss + nn.functional.cross_entropy(
                    logits[:, slot][slot_mask], move_labels[:, slot][slot_mask]
                )

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        acc, count = masked_move_accuracy(logits, move_labels, move_mask)
        total_correct += acc * count
        total_slots += count

    return total_correct / max(total_slots, 1)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_dir: Path,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = 0.0

    for epoch in range(1, epochs + 1):
        train_acc = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_acc = run_epoch(model, val_loader, device, None)
        print(f"{name} epoch {epoch:02d}  train_move={train_acc:.3f}  val_move={val_acc:.3f}")
        if val_acc >= best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": name,
                    "best_val_move_belief": best_val,
                },
                out_dir / f"{name}_best.pt",
            )

    return {"best_val_move_belief": best_val}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP/GRU move-belief baselines.")
    parser.add_argument("encoded", type=Path, nargs="+")
    parser.add_argument("--token-vocab", type=Path, default=Path("logs/vocab_belief.json"))
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/belief_baselines"))
    parser.add_argument(
        "--results-out",
        type=Path,
        default=Path("logs/belief_baseline_results.json"),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = load_encoded_records(args.encoded)
    if not any((record.get("belief") or {}).get("move_mask") for record in records):
        raise SystemExit("No active belief labels found.")

    action_vocab = ActionVocab.build_from_records(records)
    belief_vocabs = build_active_belief_vocabs(records)
    token_vocab = TokenVocab.load(args.token_vocab)

    train_tags, val_tags = ActiveBeliefDataset.split_battle_tags(records)
    train_ds = ActiveBeliefDataset.from_paths(
        args.encoded, action_vocab, belief_vocabs, args.context, train_tags
    )
    val_ds = ActiveBeliefDataset.from_paths(
        args.encoded, action_vocab, belief_vocabs, args.context, val_tags
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_moves = len(belief_vocabs["move"])

    mlp = MoveBeliefMLP(len(token_vocab), n_moves, args.d_model).to(device)
    gru = MoveBeliefGRU(len(token_vocab), n_moves, args.d_model).to(device)

    print(f"Device        {device}")
    print(f"Train samples {len(train_ds)}")
    print(f"Val samples   {len(val_ds)}")
    print(f"Move classes  {n_moves}")

    results = {
        "corpus": str(args.encoded[0]),
        "epochs": args.epochs,
        "models": {},
    }
    results["models"]["mlp"] = train_model(
        "mlp", mlp, train_loader, val_loader, device, args.epochs, args.lr, args.out_dir
    )
    results["models"]["gru"] = train_model(
        "gru", gru, train_loader, val_loader, device, args.epochs, args.lr, args.out_dir
    )
    results["comparison"] = {
        "active_belief_val_move_belief": 0.281,
        "note": "ActiveBelief reference from scaled Phase 2 run.",
    }

    args.results_out.parent.mkdir(parents=True, exist_ok=True)
    args.results_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.results_out}")


if __name__ == "__main__":
    main()

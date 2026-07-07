"""
Train a behavioral cloning model on encoded baseline logs.

Example
  python scripts/train_bc.py logs/encoded_baseline_20260529T011709Z.jsonl
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

from pokemon_thesis.data import ActionVocab, EncodedBattleDataset
from pokemon_thesis.data.encoded_dataset import load_encoded_records
from pokemon_thesis.model import TurnTransformer
from pokemon_thesis.tokenizer.vocab import TokenVocab


def train_epoch(
    model: TurnTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)

        optimizer.zero_grad()
        logits = model(token_ids, turn_mask)
        loss = loss_fn(logits, action_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * action_id.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == action_id).sum().item()
        total_samples += action_id.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


@torch.no_grad()
def eval_epoch(
    model: TurnTransformer,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)

        logits = model(token_ids, turn_mask)
        loss = loss_fn(logits, action_id)

        total_loss += loss.item() * action_id.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == action_id).sum().item()
        total_samples += action_id.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavioral cloning model.")
    parser.add_argument(
        "encoded",
        type=Path,
        nargs="+",
        help="One or more encoded JSONL files.",
    )
    parser.add_argument(
        "--token-vocab",
        type=Path,
        default=Path("logs/vocab.json"),
        help="Token vocabulary from encode_logs.py",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=8,
        help="Number of prior turns in the context window.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer hidden size.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of battles held out for validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Where to save model weights and vocab files.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Load model weights from an existing checkpoint before training.",
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
    if not records:
        raise SystemExit("No encoded records found.")

    action_vocab = ActionVocab.build_from_records(records)
    token_vocab = TokenVocab.load(args.token_vocab)

    train_tags, val_tags = EncodedBattleDataset.split_battle_tags(
        records, val_fraction=args.val_fraction
    )
    train_ds = EncodedBattleDataset.from_paths(
        args.encoded, action_vocab, context_turns=args.context, battle_tags=train_tags
    )
    val_ds = EncodedBattleDataset.from_paths(
        args.encoded, action_vocab, context_turns=args.context, battle_tags=val_tags
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TurnTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        d_model=args.d_model,
        max_context_turns=args.context,
    ).to(device)

    best_val_acc = 0.0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        if args.d_model != checkpoint.get("d_model", args.d_model):
            raise SystemExit("--d-model must match the resumed checkpoint.")
        if args.context != checkpoint.get("context_turns", args.context):
            raise SystemExit("--context must match the resumed checkpoint.")
        print(f"Resumed weights from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device            {device}")
    print(f"Train samples     {len(train_ds)}")
    print(f"Val samples       {len(val_ds)}")
    print(f"Token vocab size  {len(token_vocab)}")
    print(f"Action classes    {len(action_vocab)}")
    print(f"Context turns     {args.context}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "context_turns": args.context,
                    "d_model": args.d_model,
                    "token_vocab_size": len(token_vocab),
                    "n_actions": len(action_vocab),
                },
                args.out_dir / "bc_best.pt",
            )
            action_vocab.save(args.out_dir / "action_vocab.json")

    config = {
        "encoded_files": [str(path) for path in args.encoded],
        "token_vocab": str(args.token_vocab),
        "context_turns": args.context,
        "d_model": args.d_model,
        "best_val_acc": best_val_acc,
    }
    (args.out_dir / "bc_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Best val accuracy {best_val_acc:.3f}")
    print(f"Saved checkpoint  {args.out_dir / 'bc_best.pt'}")


if __name__ == "__main__":
    main()

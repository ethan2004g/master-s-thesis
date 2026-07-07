"""
Train belief-augmented behavioral cloning on encoded logs with belief labels.

Example
  python scripts/train_belief_bc.py logs/encoded_metamon.jsonl --epochs 30
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

from pokemon_thesis.data import ActionVocab
from pokemon_thesis.data.belief_dataset import BeliefBattleDataset
from pokemon_thesis.data.belief_labels import SpeciesBeliefVocab
from pokemon_thesis.data.encoded_dataset import load_encoded_records
from pokemon_thesis.model.belief_transformer import BeliefTransformer
from pokemon_thesis.tokenizer.vocab import TokenVocab


def species_belief_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy over opponent slots. logits shape (B, 6, C)."""
    batch_size, n_slots, n_classes = logits.shape
    flat_logits = logits.reshape(batch_size * n_slots, n_classes)
    flat_labels = labels.reshape(batch_size * n_slots)
    flat_mask = mask.reshape(batch_size * n_slots)

    valid = flat_mask > 0.5
    if not valid.any():
        return flat_logits.sum() * 0.0

    return nn.functional.cross_entropy(flat_logits[valid], flat_labels[valid])


def train_epoch(
    model: BeliefTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    action_loss_fn: nn.Module,
    device: torch.device,
    belief_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "action": 0.0, "belief": 0.0, "action_acc": 0.0}
    n_samples = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)
        species_labels = batch["species_labels"].to(device)
        species_mask = batch["species_mask"].to(device)

        optimizer.zero_grad()
        output = model(token_ids, turn_mask)
        action_loss = action_loss_fn(output.action_logits, action_id)
        belief_loss = species_belief_loss(
            output.species_logits, species_labels, species_mask
        )
        loss = action_loss + belief_weight * belief_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_n = action_id.size(0)
        n_samples += batch_n
        totals["loss"] += loss.item() * batch_n
        totals["action"] += action_loss.item() * batch_n
        totals["belief"] += belief_loss.item() * batch_n
        preds = output.action_logits.argmax(dim=-1)
        totals["action_acc"] += (preds == action_id).sum().item()

    return {key: totals[key] / max(n_samples, 1) for key in totals}


@torch.no_grad()
def eval_epoch(
    model: BeliefTransformer,
    loader: DataLoader,
    action_loss_fn: nn.Module,
    device: torch.device,
    belief_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "action": 0.0,
        "belief": 0.0,
        "action_acc": 0.0,
        "belief_acc": 0.0,
        "belief_slots": 0.0,
    }
    n_samples = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)
        species_labels = batch["species_labels"].to(device)
        species_mask = batch["species_mask"].to(device)

        output = model(token_ids, turn_mask)
        action_loss = action_loss_fn(output.action_logits, action_id)
        belief_loss = species_belief_loss(
            output.species_logits, species_labels, species_mask
        )
        loss = action_loss + belief_weight * belief_loss

        batch_n = action_id.size(0)
        n_samples += batch_n
        totals["loss"] += loss.item() * batch_n
        totals["action"] += action_loss.item() * batch_n
        totals["belief"] += belief_loss.item() * batch_n
        preds = output.action_logits.argmax(dim=-1)
        totals["action_acc"] += (preds == action_id).sum().item()

        species_preds = output.species_logits.argmax(dim=-1)
        valid = species_mask > 0.5
        if valid.any():
            correct = (species_preds[valid] == species_labels[valid]).sum().item()
            totals["belief_acc"] += correct
            totals["belief_slots"] += valid.sum().item()

    metrics = {key: totals[key] / max(n_samples, 1) for key in ("loss", "action", "belief", "action_acc")}
    if totals["belief_slots"] > 0:
        metrics["belief_acc"] = totals["belief_acc"] / totals["belief_slots"]
    else:
        metrics["belief_acc"] = 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train belief-augmented BC model.")
    parser.add_argument("encoded", type=Path, nargs="+")
    parser.add_argument("--token-vocab", type=Path, default=Path("logs/vocab.json"))
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--belief-weight", type=float, default=0.5)
    parser.add_argument("--belief-fusion", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
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

    supervised = sum(
        1 for record in records if (record.get("belief") or {}).get("opponent_species")
    )
    if supervised == 0:
        raise SystemExit(
            "No belief labels found. Encode Metamon logs with belief targets first."
        )

    action_vocab = ActionVocab.build_from_records(records)
    species_vocab = SpeciesBeliefVocab.build_from_records(records)
    token_vocab = TokenVocab.load(args.token_vocab)

    train_tags, val_tags = BeliefBattleDataset.split_battle_tags(
        records, val_fraction=args.val_fraction
    )
    train_ds = BeliefBattleDataset.from_paths(
        args.encoded,
        action_vocab,
        species_vocab,
        context_turns=args.context,
        battle_tags=train_tags,
    )
    val_ds = BeliefBattleDataset.from_paths(
        args.encoded,
        action_vocab,
        species_vocab,
        context_turns=args.context,
        battle_tags=val_tags,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = BeliefTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        n_species=len(species_vocab),
        d_model=args.d_model,
        max_context_turns=args.context,
        belief_fusion=args.belief_fusion,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    action_loss_fn = nn.CrossEntropyLoss()

    print(f"Device                 {device}")
    print(f"Train samples          {len(train_ds)}")
    print(f"Val samples            {len(val_ds)}")
    print(f"Supervised species slots (train) {train_ds.count_supervised_species_slots()}")
    print(f"Species classes        {len(species_vocab)}")
    print(f"Action classes         {len(action_vocab)}")
    print(f"Belief weight          {args.belief_weight}")

    best_val_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            action_loss_fn,
            device,
            args.belief_weight,
        )
        val_metrics = eval_epoch(
            model,
            val_loader,
            action_loss_fn,
            device,
            args.belief_weight,
        )
        print(
            f"Epoch {epoch:02d}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_action_acc={train_metrics['action_acc']:.3f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_action_acc={val_metrics['action_acc']:.3f}  "
            f"val_belief_acc={val_metrics['belief_acc']:.3f}"
        )
        if val_metrics["action_acc"] >= best_val_acc:
            best_val_acc = val_metrics["action_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "belief_transformer",
                    "context_turns": args.context,
                    "d_model": args.d_model,
                    "token_vocab_size": len(token_vocab),
                    "n_actions": len(action_vocab),
                    "n_species": len(species_vocab),
                    "belief_fusion": args.belief_fusion,
                },
                args.out_dir / "belief_bc_best.pt",
            )
            action_vocab.save(args.out_dir / "action_vocab.json")
            species_vocab.save(args.out_dir / "species_belief_vocab.json")

    config = {
        "encoded_files": [str(path) for path in args.encoded],
        "belief_weight": args.belief_weight,
        "belief_fusion": args.belief_fusion,
        "best_val_action_acc": best_val_acc,
    }
    (args.out_dir / "belief_bc_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Best val action accuracy {best_val_acc:.3f}")
    print(f"Saved checkpoint        {args.out_dir / 'belief_bc_best.pt'}")


if __name__ == "__main__":
    main()

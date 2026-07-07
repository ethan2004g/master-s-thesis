"""
Train active-opponent belief + matchup-conditioned BC.

Example
  python scripts/train_active_belief.py logs/encoded_belief_corpus.jsonl --epochs 20
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
from pokemon_thesis.model.active_belief_transformer import ActiveBeliefTransformer
from pokemon_thesis.tokenizer.vocab import TokenVocab


def masked_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_mask = mask.reshape(-1) > 0.5
    if not flat_mask.any():
        return flat_logits.sum() * 0.0
    return nn.functional.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train active belief transformer.")
    parser.add_argument("encoded", type=Path, nargs="+")
    parser.add_argument("--token-vocab", type=Path, default=Path("logs/vocab_belief.json"))
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--belief-weight", type=float, default=0.5)
    parser.add_argument(
        "--belief-fusion",
        action="store_true",
        help="Concatenate belief softmax outputs to the policy head input.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/active_belief"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run_epoch(
    model: ActiveBeliefTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    action_loss_fn: nn.Module,
    belief_weight: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals = {"action_acc": 0.0, "belief_acc": 0.0, "belief_count": 0.0, "n": 0.0}

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)
        player_ids = batch["player_species_ids"].to(device)
        opponent_ids = batch["opponent_species_ids"].to(device)

        if train_mode:
            optimizer.zero_grad()
        output = model(token_ids, turn_mask, player_ids, opponent_ids)
        action_loss = action_loss_fn(output.action_logits, action_id)
        move_loss = masked_ce(output.move_logits, batch["move_labels"].to(device), batch["move_mask"].to(device))
        item_loss = masked_ce(
            output.item_logits,
            batch["item_label"].to(device),
            batch["item_mask"].to(device).unsqueeze(-1),
        )
        ability_loss = masked_ce(
            output.ability_logits,
            batch["ability_label"].to(device),
            batch["ability_mask"].to(device).unsqueeze(-1),
        )
        tera_loss = masked_ce(
            output.tera_logits,
            batch["tera_label"].to(device),
            batch["tera_mask"].to(device).unsqueeze(-1),
        )
        belief_loss = move_loss + item_loss + ability_loss + tera_loss
        loss = action_loss + belief_weight * belief_loss

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_n = action_id.size(0)
        totals["n"] += batch_n
        totals["action_acc"] += (output.action_logits.argmax(-1) == action_id).sum().item()

        move_preds = output.move_logits.argmax(-1)
        move_mask = batch["move_mask"].to(device) > 0.5
        if move_mask.any():
            totals["belief_acc"] += (move_preds[move_mask] == batch["move_labels"].to(device)[move_mask]).sum().item()
            totals["belief_count"] += move_mask.sum().item()

    metrics = {"action_acc": totals["action_acc"] / max(totals["n"], 1)}
    metrics["belief_acc"] = totals["belief_acc"] / max(totals["belief_count"], 1)
    return metrics


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = load_encoded_records(args.encoded)
    if not any((record.get("belief") or {}).get("move_mask") for record in records):
        raise SystemExit("No active belief labels found. Run export_belief_corpus.py first.")

    action_vocab = ActionVocab.build_from_records(records)
    belief_vocabs = build_active_belief_vocabs(records)
    token_vocab = TokenVocab.load(args.token_vocab)

    train_tags, val_tags = ActiveBeliefDataset.split_battle_tags(
        records, val_fraction=args.val_fraction
    )
    train_ds = ActiveBeliefDataset.from_paths(
        args.encoded, action_vocab, belief_vocabs, args.context, train_tags
    )
    val_ds = ActiveBeliefDataset.from_paths(
        args.encoded, action_vocab, belief_vocabs, args.context, val_tags
    )
    print(f"Supervised train counts {train_ds.supervised_counts()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ActiveBeliefTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        n_species=len(belief_vocabs["species"]),
        n_moves=len(belief_vocabs["move"]),
        n_items=len(belief_vocabs["item"]),
        n_abilities=len(belief_vocabs["ability"]),
        n_tera=len(belief_vocabs["tera"]),
        d_model=args.d_model,
        max_context_turns=args.context,
        belief_fusion=args.belief_fusion,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    action_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, optimizer, action_loss_fn, args.belief_weight
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, device, None, action_loss_fn, args.belief_weight
            )
        print(
            f"Epoch {epoch:02d}  train_action={train_metrics['action_acc']:.3f}  "
            f"val_action={val_metrics['action_acc']:.3f}  "
            f"val_move_belief={val_metrics['belief_acc']:.3f}"
        )
        if val_metrics["action_acc"] >= best_val_acc:
            best_val_acc = val_metrics["action_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "active_belief_transformer",
                    "belief_fusion": args.belief_fusion,
                    "context_turns": args.context,
                    "d_model": args.d_model,
                    "token_vocab_size": len(token_vocab),
                    "n_actions": len(action_vocab),
                    "n_species": len(belief_vocabs["species"]),
                    "n_moves": len(belief_vocabs["move"]),
                    "n_items": len(belief_vocabs["item"]),
                    "n_abilities": len(belief_vocabs["ability"]),
                    "n_tera": len(belief_vocabs["tera"]),
                },
                args.out_dir / "active_belief_best.pt",
            )
            action_vocab.save(args.out_dir / "action_vocab.json")
            for name, vocab in belief_vocabs.items():
                vocab.save(args.out_dir / f"{name}_vocab.json")

    (args.out_dir / "active_belief_config.json").write_text(
        json.dumps(
            {
                "best_val_action_acc": best_val_acc,
                "belief_fusion": args.belief_fusion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

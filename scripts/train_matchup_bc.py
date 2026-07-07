"""
Train matchup-conditioned BC (team composition ablation).

Example
  python scripts/train_matchup_bc.py logs/encoded_belief_corpus.jsonl --epochs 20
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
from pokemon_thesis.model.active_belief_transformer import MatchupTurnTransformer
from pokemon_thesis.tokenizer.vocab import TokenVocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train matchup-conditioned BC.")
    parser.add_argument("encoded", type=Path, nargs="+")
    parser.add_argument("--token-vocab", type=Path, default=Path("logs/vocab_belief.json"))
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/matchup_bc"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = load_encoded_records(args.encoded)
    if not any(record.get("team_matchup") for record in records):
        raise SystemExit("No team_matchup fields found. Run export_belief_corpus.py first.")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MatchupTurnTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        n_species=len(belief_vocabs["species"]),
        d_model=args.d_model,
        max_context_turns=args.context,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device         {device}")
    print(f"Train samples  {len(train_ds)}")
    print(f"Val samples    {len(val_ds)}")

    best_val_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            turn_mask = batch["turn_mask"].to(device)
            action_id = batch["action_id"].to(device)
            player_ids = batch["player_species_ids"].to(device)
            opponent_ids = batch["opponent_species_ids"].to(device)

            optimizer.zero_grad()
            logits = model(token_ids, turn_mask, player_ids, opponent_ids)
            loss = loss_fn(logits, action_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * action_id.size(0)
            train_correct += (logits.argmax(-1) == action_id).sum().item()
            train_total += action_id.size(0)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch["token_ids"].to(device)
                turn_mask = batch["turn_mask"].to(device)
                action_id = batch["action_id"].to(device)
                player_ids = batch["player_species_ids"].to(device)
                opponent_ids = batch["opponent_species_ids"].to(device)
                logits = model(token_ids, turn_mask, player_ids, opponent_ids)
                loss = loss_fn(logits, action_id)
                val_loss_sum += loss.item() * action_id.size(0)
                val_correct += (logits.argmax(-1) == action_id).sum().item()
                val_total += action_id.size(0)

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch:02d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
            f"val_loss={val_loss_sum / max(val_total, 1):.4f}"
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": "matchup_turn_transformer",
                    "context_turns": args.context,
                    "d_model": args.d_model,
                    "token_vocab_size": len(token_vocab),
                    "n_actions": len(action_vocab),
                    "n_species": len(belief_vocabs["species"]),
                },
                args.out_dir / "matchup_bc_best.pt",
            )
            action_vocab.save(args.out_dir / "action_vocab.json")
            belief_vocabs["species"].save(args.out_dir / "species_vocab.json")

    (args.out_dir / "matchup_bc_config.json").write_text(
        json.dumps({"best_val_acc": best_val_acc}, indent=2), encoding="utf-8"
    )
    print(f"Best val accuracy {best_val_acc:.3f}")


if __name__ == "__main__":
    main()

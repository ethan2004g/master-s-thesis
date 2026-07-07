"""
Train linear probes on frozen TurnTransformer trunk embeddings for move belief.

Compares probe accuracy to ActiveBeliefTransformer move heads on the same val split.

Example
  python scripts/probe_trunk_belief.py
  python scripts/probe_trunk_belief.py --epochs 15 --checkpoint-dir checkpoints/human_bc_belief
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
from pokemon_thesis.data.active_belief_labels import BeliefLabelVocab, build_active_belief_vocabs
from pokemon_thesis.data.encoded_dataset import load_encoded_records
from pokemon_thesis.model import TurnTransformer
from pokemon_thesis.model.active_belief_transformer import ActiveBeliefTransformer, MOVE_SLOTS
from pokemon_thesis.tokenizer.vocab import TokenVocab


class MoveSlotProbe(nn.Module):
    def __init__(self, d_model: int, n_moves: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Linear(d_model, n_moves) for _ in range(MOVE_SLOTS)]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(embedding) for head in self.heads], dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear trunk probes for move belief.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("logs/encoded_belief_corpus.jsonl"),
    )
    parser.add_argument(
        "--token-vocab",
        type=Path,
        default=Path("logs/vocab_belief.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/human_bc_belief"),
    )
    parser.add_argument(
        "--active-checkpoint-dir",
        type=Path,
        default=Path("checkpoints/active_belief"),
    )
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("logs/trunk_probe_results.json"),
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_turn_model(
    checkpoint_dir: Path, token_vocab: TokenVocab, device: torch.device
) -> TurnTransformer:
    checkpoint = torch.load(
        checkpoint_dir / "bc_best.pt", map_location=device, weights_only=True
    )
    action_vocab = ActionVocab.load(checkpoint_dir / "action_vocab.json")
    model = TurnTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        d_model=checkpoint["d_model"],
        max_context_turns=checkpoint["context_turns"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


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


def train_probe_epoch(
    trunk: TurnTransformer,
    probe: MoveSlotProbe,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    probe.train()
    total_loss = 0.0
    total_correct = 0
    total_slots = 0

    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        move_labels = batch["move_labels"].to(device)
        move_mask = batch["move_mask"].to(device)

        with torch.no_grad():
            embedding = trunk.encode_final_turn(token_ids, turn_mask)

        optimizer.zero_grad()
        logits = probe(embedding)
        loss = 0.0
        for slot in range(MOVE_SLOTS):
            slot_mask = move_mask[:, slot] > 0
            if slot_mask.any():
                loss = loss + loss_fn(logits[:, slot][slot_mask], move_labels[:, slot][slot_mask])
        loss.backward()
        optimizer.step()

        batch_size = token_ids.size(0)
        total_loss += loss.item() * batch_size
        acc, count = masked_move_accuracy(logits, move_labels, move_mask)
        total_correct += acc * count
        total_slots += count

    return total_loss / max(len(loader.dataset), 1), total_correct / max(total_slots, 1)


@torch.no_grad()
def eval_probe(
    trunk: TurnTransformer,
    probe: MoveSlotProbe,
    loader: DataLoader,
    device: torch.device,
) -> float:
    probe.eval()
    total_correct = 0
    total_slots = 0
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        move_labels = batch["move_labels"].to(device)
        move_mask = batch["move_mask"].to(device)
        embedding = trunk.encode_final_turn(token_ids, turn_mask)
        logits = probe(embedding)
        acc, count = masked_move_accuracy(logits, move_labels, move_mask)
        total_correct += acc * count
        total_slots += count
    return total_correct / max(total_slots, 1)


@torch.no_grad()
def eval_active_move_belief(
    checkpoint_dir: Path,
    token_vocab: TokenVocab,
    belief_vocabs: dict[str, BeliefLabelVocab],
    action_vocab: ActionVocab,
    loader: DataLoader,
    device: torch.device,
) -> float:
    checkpoint = torch.load(
        checkpoint_dir / "active_belief_best.pt", map_location=device, weights_only=True
    )
    for key in ("move", "item", "ability", "tera", "species"):
        path = checkpoint_dir / f"{key}_vocab.json"
        if path.is_file():
            belief_vocabs[key] = BeliefLabelVocab.load(path)

    model = ActiveBeliefTransformer(
        token_vocab_size=len(token_vocab),
        n_actions=len(action_vocab),
        n_species=len(belief_vocabs["species"]),
        n_moves=len(belief_vocabs["move"]),
        n_items=len(belief_vocabs["item"]),
        n_abilities=len(belief_vocabs["ability"]),
        n_tera=len(belief_vocabs["tera"]),
        d_model=checkpoint["d_model"],
        max_context_turns=checkpoint["context_turns"],
        belief_fusion=checkpoint.get("belief_fusion", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_correct = 0
    total_slots = 0
    for batch in loader:
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        move_labels = batch["move_labels"].to(device)
        move_mask = batch["move_mask"].to(device)
        player_ids = batch["player_species_ids"].to(device)
        opponent_ids = batch["opponent_species_ids"].to(device)
        output = model(token_ids, turn_mask, player_ids, opponent_ids)
        acc, count = masked_move_accuracy(output.move_logits, move_labels, move_mask)
        total_correct += acc * count
        total_slots += count
    return total_correct / max(total_slots, 1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    records = load_encoded_records([args.corpus])
    action_vocab = ActionVocab.build_from_records(records)
    belief_vocabs = build_active_belief_vocabs(records)
    token_vocab = TokenVocab.load(args.token_vocab)

    train_tags, val_tags = ActiveBeliefDataset.split_battle_tags(records)
    train_ds = ActiveBeliefDataset.from_paths(
        [args.corpus], action_vocab, belief_vocabs, args.context, train_tags
    )
    val_ds = ActiveBeliefDataset.from_paths(
        [args.corpus], action_vocab, belief_vocabs, args.context, val_tags
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    trunk = load_turn_model(args.checkpoint_dir, token_vocab, device)
    probe = MoveSlotProbe(
        d_model=trunk.d_model, n_moves=len(belief_vocabs["move"])
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device          {device}")
    print(f"Train samples   {len(train_ds)}")
    print(f"Val samples     {len(val_ds)}")
    print(f"Move classes    {len(belief_vocabs['move'])}")

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_probe_epoch(
            trunk, probe, train_loader, optimizer, loss_fn, device
        )
        val_acc = eval_probe(trunk, probe, val_loader, device)
        print(
            f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  "
            f"train_move={train_acc:.3f}  val_move={val_acc:.3f}"
        )
        best_val = max(best_val, val_acc)

    active_val = eval_active_move_belief(
        args.active_checkpoint_dir,
        token_vocab,
        belief_vocabs,
        action_vocab,
        val_loader,
        device,
    )

    results = {
        "corpus": str(args.corpus),
        "trunk_checkpoint": str(args.checkpoint_dir),
        "active_checkpoint": str(args.active_checkpoint_dir),
        "probe_epochs": args.epochs,
        "val_move_belief_probe": best_val,
        "val_move_belief_active_heads": active_val,
        "interpretation": (
            "If probe accuracy is close to active belief heads, the BC trunk "
            "already linearly encodes move information."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

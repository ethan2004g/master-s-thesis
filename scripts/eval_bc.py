"""
Evaluate BC checkpoints on in-distribution and held-out team splits.

Supports TurnTransformer (human_bc) and MatchupTurnTransformer (matchup_bc).

Example
  python scripts/eval_bc.py --checkpoint-dir checkpoints/human_bc --corpus logs/encoded_belief_corpus.jsonl --token-vocab logs/vocab_belief.json
  python scripts/eval_bc.py --checkpoint-dir checkpoints/matchup_bc --corpus logs/encoded_belief_corpus.jsonl --token-vocab logs/vocab_belief.json
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
sys.path.insert(0, str(ROOT))

from pokemon_thesis.data import ActionVocab, EncodedBattleDataset
from pokemon_thesis.data.active_belief_dataset import ActiveBeliefDataset
from pokemon_thesis.data.active_belief_labels import BeliefLabelVocab, build_active_belief_vocabs
from pokemon_thesis.data.encoded_dataset import load_encoded_records
from pokemon_thesis.data.splits import held_out_team_split
from pokemon_thesis.model import TurnTransformer
from pokemon_thesis.model.active_belief_transformer import (
    ActiveBeliefTransformer,
    MatchupTurnTransformer,
)
from pokemon_thesis.tokenizer.vocab import TokenVocab
from scripts.train_bc import eval_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BC on ID and OOD splits.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("logs/encoded_belief_corpus.jsonl"),
        help="Encoded corpus with team_matchup fields.",
    )
    parser.add_argument(
        "--encoded",
        type=Path,
        default=None,
        help="Deprecated alias for --corpus.",
    )
    parser.add_argument(
        "--ood-encoded",
        type=Path,
        default=None,
        help="Deprecated alias for --corpus.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/human_bc"),
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Override checkpoint filename (default auto-detect).",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "turn", "matchup", "active"),
        default="auto",
    )
    parser.add_argument(
        "--token-vocab",
        type=Path,
        default=Path("logs/vocab_belief.json"),
    )
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default derived from checkpoint dir).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_corpus(args: argparse.Namespace) -> Path:
    if args.corpus and args.corpus != Path("logs/encoded_belief_corpus.jsonl"):
        return args.corpus
    if args.ood_encoded:
        return args.ood_encoded
    if args.encoded:
        return args.encoded
    return args.corpus


def resolve_checkpoint_file(checkpoint_dir: Path, override: str | None) -> Path:
    if override:
        return checkpoint_dir / override
    for name in ("matchup_bc_best.pt", "bc_best.pt", "active_belief_best.pt"):
        path = checkpoint_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def resolve_model_type(
    args: argparse.Namespace, checkpoint: dict, checkpoint_path: Path
) -> str:
    if args.model_type != "auto":
        return args.model_type
    if checkpoint_path.name == "active_belief_best.pt":
        return "active"
    if checkpoint.get("model_type") == "active_belief_transformer":
        return "active"
    if checkpoint.get("model_type") == "matchup_turn_transformer":
        return "matchup"
    if checkpoint_path.name == "matchup_bc_best.pt":
        return "matchup"
    return "turn"


def eval_turn_split(
    model: TurnTransformer,
    records_path: Path,
    action_vocab: ActionVocab,
    battle_tags: set[str],
    context: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataset = EncodedBattleDataset.from_paths(
        [records_path], action_vocab, context_turns=context, battle_tags=battle_tags
    )
    if len(dataset) == 0:
        return {"samples": 0, "loss": 0.0, "accuracy": 0.0}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()
    loss, acc = eval_epoch(model, loader, loss_fn, device)
    return {"samples": len(dataset), "loss": loss, "accuracy": acc}


@torch.no_grad()
def eval_matchup_split(
    model: MatchupTurnTransformer,
    records_path: Path,
    action_vocab: ActionVocab,
    belief_vocabs: dict[str, BeliefLabelVocab],
    battle_tags: set[str],
    context: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataset = ActiveBeliefDataset.from_paths(
        [records_path],
        action_vocab,
        belief_vocabs,
        context_turns=context,
        battle_tags=battle_tags,
    )
    if len(dataset) == 0:
        return {"samples": 0, "loss": 0.0, "accuracy": 0.0}

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)
        player_ids = batch["player_species_ids"].to(device)
        opponent_ids = batch["opponent_species_ids"].to(device)
        logits = model(token_ids, turn_mask, player_ids, opponent_ids)
        loss = loss_fn(logits, action_id)
        total_loss += loss.item() * action_id.size(0)
        total_correct += (logits.argmax(-1) == action_id).sum().item()
        total_samples += action_id.size(0)

    return {
        "samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def eval_active_split(
    model: ActiveBeliefTransformer,
    records_path: Path,
    action_vocab: ActionVocab,
    belief_vocabs: dict[str, BeliefLabelVocab],
    battle_tags: set[str],
    context: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    dataset = ActiveBeliefDataset.from_paths(
        [records_path],
        action_vocab,
        belief_vocabs,
        context_turns=context,
        battle_tags=battle_tags,
    )
    if len(dataset) == 0:
        return {"samples": 0, "loss": 0.0, "accuracy": 0.0}

    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        token_ids = batch["token_ids"].to(device)
        turn_mask = batch["turn_mask"].to(device)
        action_id = batch["action_id"].to(device)
        player_ids = batch["player_species_ids"].to(device)
        opponent_ids = batch["opponent_species_ids"].to(device)
        output = model(token_ids, turn_mask, player_ids, opponent_ids)
        loss = loss_fn(output.action_logits, action_id)
        total_loss += loss.item() * action_id.size(0)
        total_correct += (output.action_logits.argmax(-1) == action_id).sum().item()
        total_samples += action_id.size(0)

    return {
        "samples": total_samples,
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    corpus_path = resolve_corpus(args)
    records = load_encoded_records([corpus_path])

    if not any(record.get("team_matchup") for record in records):
        raise SystemExit(f"{corpus_path} has no team_matchup fields. Run export_belief_corpus.py.")

    checkpoint_path = resolve_checkpoint_file(args.checkpoint_dir, args.checkpoint_file)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type = resolve_model_type(args, checkpoint, checkpoint_path)

    action_vocab = ActionVocab.load(args.checkpoint_dir / "action_vocab.json")
    token_vocab = TokenVocab.load(args.token_vocab)

    _, id_val_tags = EncodedBattleDataset.split_battle_tags(records, val_fraction=0.2)
    id_train_tags, ood_tags = held_out_team_split(records, holdout_fraction=0.2)

    belief_vocabs = build_active_belief_vocabs(records)

    if model_type == "active":
        for key in ("move", "item", "ability", "tera", "species"):
            vocab_path = args.checkpoint_dir / f"{key}_vocab.json"
            if vocab_path.is_file():
                belief_vocabs[key] = BeliefLabelVocab.load(vocab_path)
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
        eval_fn = lambda tags: eval_active_split(
            model,
            corpus_path,
            action_vocab,
            belief_vocabs,
            tags,
            args.context,
            args.batch_size,
            device,
        )
    elif model_type == "matchup":
        species_vocab = BeliefLabelVocab.load(args.checkpoint_dir / "species_vocab.json")
        belief_vocabs["species"] = species_vocab
        model = MatchupTurnTransformer(
            token_vocab_size=len(token_vocab),
            n_actions=len(action_vocab),
            n_species=len(species_vocab),
            d_model=checkpoint["d_model"],
            max_context_turns=checkpoint["context_turns"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        eval_fn = lambda tags: eval_matchup_split(
            model,
            corpus_path,
            action_vocab,
            belief_vocabs,
            tags,
            args.context,
            args.batch_size,
            device,
        )
    else:
        model = TurnTransformer(
            token_vocab_size=len(token_vocab),
            n_actions=len(action_vocab),
            d_model=checkpoint["d_model"],
            max_context_turns=checkpoint["context_turns"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        eval_fn = lambda tags: eval_turn_split(
            model,
            corpus_path,
            action_vocab,
            tags,
            args.context,
            args.batch_size,
            device,
        )

    results = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_file": checkpoint_path.name,
        "model_type": model_type,
        "corpus": str(corpus_path),
        "id_val_split": eval_fn(id_val_tags),
        "ood_team_split": eval_fn(ood_tags),
        "id_train_team_split": eval_fn(id_train_tags),
        "ood_team_keys_held_out": len(ood_tags),
        "id_train_team_keys": len(id_train_tags),
    }
    if results["id_val_split"]["accuracy"] > 0 and results["ood_team_split"]["samples"] > 0:
        results["generalization_gap"] = (
            results["id_val_split"]["accuracy"] - results["ood_team_split"]["accuracy"]
        )

    out_path = args.out or Path(f"logs/eval_{args.checkpoint_dir.name}_ood.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

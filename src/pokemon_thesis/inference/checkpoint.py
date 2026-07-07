from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from pokemon_thesis.data.action_vocab import ActionVocab
from pokemon_thesis.model import TurnTransformer
from pokemon_thesis.tokenizer.vocab import TokenVocab


@dataclass
class BCCheckpoint:
    model: TurnTransformer
    token_vocab: TokenVocab
    action_vocab: ActionVocab
    context_turns: int
    device: torch.device


def load_bc_checkpoint(
    checkpoint_dir: Path,
    device: torch.device | None = None,
) -> BCCheckpoint:
    checkpoint_dir = checkpoint_dir.resolve()
    weights_path = checkpoint_dir / "bc_best.pt"
    action_vocab_path = checkpoint_dir / "action_vocab.json"
    config_path = checkpoint_dir / "bc_config.json"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing {weights_path}. Train with scripts/train_bc.py first."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    token_vocab_path = Path(config.get("token_vocab", "logs/vocab.json"))
    if not token_vocab_path.is_absolute():
        token_vocab_path = (checkpoint_dir.parent / token_vocab_path).resolve()

    resolved_device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    context_turns = int(config.get("context_turns", 8))
    d_model = int(config.get("d_model", 128))

    token_vocab = TokenVocab.load(token_vocab_path)
    action_vocab = ActionVocab.load(action_vocab_path)

    payload = torch.load(weights_path, map_location=resolved_device, weights_only=True)
    model = TurnTransformer(
        token_vocab_size=int(payload["token_vocab_size"]),
        n_actions=int(payload["n_actions"]),
        d_model=d_model,
        max_context_turns=context_turns,
    )
    model.load_state_dict(payload["model_state"])
    model.to(resolved_device)
    model.eval()

    return BCCheckpoint(
        model=model,
        token_vocab=token_vocab,
        action_vocab=action_vocab,
        context_turns=context_turns,
        device=resolved_device,
    )

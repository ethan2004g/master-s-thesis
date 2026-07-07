from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from pokemon_thesis.model.active_belief_transformer import MOVE_SLOTS
from pokemon_thesis.tokenizer.turn_tokenizer import TURN_TOKEN_COUNT


class MoveBeliefMLP(nn.Module):
    """Predict active-opponent moves from the latest turn tokens only."""

    def __init__(
        self,
        token_vocab_size: int,
        n_moves: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(token_vocab_size, d_model, padding_idx=0)
        self.trunk = nn.Sequential(
            nn.Linear(TURN_TOKEN_COUNT * d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
        )
        self.move_heads = nn.ModuleList(
            [nn.Linear(d_model, n_moves) for _ in range(MOVE_SLOTS)]
        )

    def encode(self, token_ids: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        del turn_mask
        latest = token_ids[:, -1, :]
        emb = self.token_embed(latest)
        flat = emb.reshape(emb.size(0), -1)
        return self.trunk(flat)

    def forward(self, token_ids: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(token_ids, turn_mask)
        return torch.stack([head(hidden) for head in self.move_heads], dim=1)


class MoveBeliefGRU(nn.Module):
    """Predict active-opponent moves from a GRU over turn embeddings."""

    def __init__(
        self,
        token_vocab_size: int,
        n_moves: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(token_vocab_size, d_model, padding_idx=0)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.move_heads = nn.ModuleList(
            [nn.Linear(d_model, n_moves) for _ in range(MOVE_SLOTS)]
        )

    def encode(self, token_ids: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        batch_size, n_turns, _ = token_ids.shape
        flat_tokens = token_ids.reshape(batch_size * n_turns, TURN_TOKEN_COUNT)
        slot_emb = self.token_embed(flat_tokens).mean(dim=1)
        turn_emb = slot_emb.reshape(batch_size, n_turns, self.d_model)
        lengths = turn_mask.sum(dim=1).long().clamp(min=1)
        packed = pack_padded_sequence(
            turn_emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]

    def forward(self, token_ids: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(token_ids, turn_mask)
        return torch.stack([head(hidden) for head in self.move_heads], dim=1)

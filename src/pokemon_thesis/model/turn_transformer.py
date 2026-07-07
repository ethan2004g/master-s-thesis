from __future__ import annotations

import torch
import torch.nn as nn

from pokemon_thesis.tokenizer.turn_tokenizer import TURN_TOKEN_COUNT


class TurnTransformer(nn.Module):
    """
    Transformer for behavioral cloning on encoded turns.

    For each turn, self-attention runs across the 13 tokens.
    Across the battle, causal attention runs across turn embeddings.
    """

    def __init__(
        self,
        token_vocab_size: int,
        n_actions: int,
        d_model: int = 128,
        n_heads: int = 4,
        turn_layers: int = 2,
        slot_layers: int = 1,
        dropout: float = 0.1,
        max_context_turns: int = 32,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_context_turns = max_context_turns

        self.token_embed = nn.Embedding(token_vocab_size, d_model, padding_idx=0)
        self.slot_pos_embed = nn.Embedding(TURN_TOKEN_COUNT, d_model)
        self.turn_pos_embed = nn.Embedding(max_context_turns, d_model)

        slot_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(slot_layer, num_layers=slot_layers)

        turn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.turn_encoder = nn.TransformerEncoder(turn_layer, num_layers=turn_layers)
        self.pad_turn_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.action_head = nn.Linear(d_model, n_actions)

    def encode_final_turn(
        self,
        token_ids: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the final valid turn embedding for each batch row."""
        batch_size, n_turns, _ = token_ids.shape
        if n_turns > self.max_context_turns:
            raise ValueError(
                f"context has {n_turns} turns but max_context_turns is "
                f"{self.max_context_turns}"
            )

        slot_positions = torch.arange(
            TURN_TOKEN_COUNT, device=token_ids.device
        ).unsqueeze(0)
        turn_positions = torch.arange(n_turns, device=token_ids.device).unsqueeze(0)

        flat_tokens = token_ids.reshape(batch_size * n_turns, TURN_TOKEN_COUNT)
        slot_emb = self.token_embed(flat_tokens) + self.slot_pos_embed(slot_positions)
        slot_encoded = self.slot_encoder(slot_emb)
        turn_emb = slot_encoded.mean(dim=1).reshape(batch_size, n_turns, self.d_model)
        turn_emb = turn_emb + self.turn_pos_embed(turn_positions)

        pad_mask = (turn_mask == 0).unsqueeze(-1)
        turn_emb = torch.where(pad_mask, self.pad_turn_embed, turn_emb)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            n_turns, device=token_ids.device, dtype=turn_emb.dtype
        )
        turn_encoded = self.turn_encoder(turn_emb, mask=causal_mask)

        valid_lengths = turn_mask.sum(dim=1).long().clamp(min=1)
        last_index = valid_lengths - 1
        batch_indices = torch.arange(batch_size, device=token_ids.device)
        return turn_encoded[batch_indices, last_index]

    def forward(
        self,
        token_ids: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args
            token_ids  (B, T, 13) integer token ids
            turn_mask  (B, T)     1 for real turns, 0 for left padding

        Returns
            logits (B, n_actions)
        """
        final_turn = self.encode_final_turn(token_ids, turn_mask)
        return self.action_head(final_turn)

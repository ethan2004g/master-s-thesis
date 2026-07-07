from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pokemon_thesis.model.turn_transformer import TurnTransformer
from pokemon_thesis.tokenizer.turn_tokenizer import PARTY_SIZE


@dataclass(frozen=True)
class BeliefTransformerOutput:
    action_logits: torch.Tensor
    species_logits: torch.Tensor


class BeliefTransformer(TurnTransformer):
    """
    TurnTransformer with per-slot opponent species belief heads.

    Optional belief fusion concatenates species softmax outputs to the policy
    input for ablation studies.
    """

    def __init__(
        self,
        token_vocab_size: int,
        n_actions: int,
        n_species: int,
        d_model: int = 128,
        n_heads: int = 4,
        turn_layers: int = 2,
        slot_layers: int = 1,
        dropout: float = 0.1,
        max_context_turns: int = 32,
        belief_fusion: bool = False,
    ) -> None:
        super().__init__(
            token_vocab_size=token_vocab_size,
            n_actions=n_actions,
            d_model=d_model,
            n_heads=n_heads,
            turn_layers=turn_layers,
            slot_layers=slot_layers,
            dropout=dropout,
            max_context_turns=max_context_turns,
        )
        self.n_species = n_species
        self.belief_fusion = belief_fusion

        self.species_heads = nn.ModuleList(
            [nn.Linear(d_model, n_species) for _ in range(PARTY_SIZE)]
        )

        if belief_fusion:
            fusion_dim = d_model + PARTY_SIZE * n_species
            self.action_head = nn.Linear(fusion_dim, n_actions)
        else:
            self.action_head = nn.Linear(d_model, n_actions)

    def species_logits(self, final_turn: torch.Tensor) -> torch.Tensor:
        """Return species logits with shape (B, 6, n_species)."""
        slot_logits = [head(final_turn) for head in self.species_heads]
        return torch.stack(slot_logits, dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        turn_mask: torch.Tensor,
    ) -> BeliefTransformerOutput:
        final_turn = self.encode_final_turn(token_ids, turn_mask)
        species = self.species_logits(final_turn)

        if self.belief_fusion:
            species_probs = torch.softmax(species, dim=-1)
            fused = torch.cat(
                [final_turn, species_probs.reshape(species.size(0), -1)],
                dim=-1,
            )
            action_logits = self.action_head(fused)
        else:
            action_logits = self.action_head(final_turn)

        return BeliefTransformerOutput(
            action_logits=action_logits,
            species_logits=species,
        )

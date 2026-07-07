from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from pokemon_thesis.model.team_matchup_encoder import TeamMatchupEncoder
from pokemon_thesis.model.turn_transformer import TurnTransformer

MOVE_SLOTS = 4


@dataclass(frozen=True)
class ActiveBeliefOutput:
    action_logits: torch.Tensor
    move_logits: torch.Tensor
    item_logits: torch.Tensor
    ability_logits: torch.Tensor
    tera_logits: torch.Tensor


class MatchupTurnTransformer(TurnTransformer):
    """TurnTransformer with team matchup conditioning on the final turn embedding."""

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
        self.matchup_encoder = TeamMatchupEncoder(n_species, d_model)

    def forward(
        self,
        token_ids: torch.Tensor,
        turn_mask: torch.Tensor,
        player_species_ids: torch.Tensor,
        opponent_species_ids: torch.Tensor,
    ) -> torch.Tensor:
        final_turn = self.encode_final_turn(token_ids, turn_mask)
        matchup = self.matchup_encoder(player_species_ids, opponent_species_ids)
        return self.action_head(final_turn + matchup)


class ActiveBeliefTransformer(MatchupTurnTransformer):
    """Matchup-conditioned policy with active-opponent belief heads."""

    def __init__(
        self,
        token_vocab_size: int,
        n_actions: int,
        n_species: int,
        n_moves: int,
        n_items: int,
        n_abilities: int,
        n_tera: int,
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
            n_species=n_species,
            d_model=d_model,
            n_heads=n_heads,
            turn_layers=turn_layers,
            slot_layers=slot_layers,
            dropout=dropout,
            max_context_turns=max_context_turns,
        )
        self.belief_fusion = belief_fusion
        self.move_heads = nn.ModuleList(
            [nn.Linear(d_model, n_moves) for _ in range(MOVE_SLOTS)]
        )
        self.item_head = nn.Linear(d_model, n_items)
        self.ability_head = nn.Linear(d_model, n_abilities)
        self.tera_head = nn.Linear(d_model, n_tera)

        if belief_fusion:
            belief_prob_dim = MOVE_SLOTS * n_moves + n_items + n_abilities + n_tera
            self.action_head = nn.Linear(d_model + belief_prob_dim, n_actions)

    def _policy_input(
        self,
        fused: torch.Tensor,
        move_logits: torch.Tensor,
        item_logits: torch.Tensor,
        ability_logits: torch.Tensor,
        tera_logits: torch.Tensor,
    ) -> torch.Tensor:
        if not self.belief_fusion:
            return fused
        move_probs = torch.softmax(move_logits, dim=-1).reshape(move_logits.size(0), -1)
        item_probs = torch.softmax(item_logits, dim=-1)
        ability_probs = torch.softmax(ability_logits, dim=-1)
        tera_probs = torch.softmax(tera_logits, dim=-1)
        return torch.cat(
            [fused, move_probs, item_probs, ability_probs, tera_probs],
            dim=-1,
        )

    def belief_logits(self, fused: torch.Tensor) -> tuple[torch.Tensor, ...]:
        move_logits = torch.stack([head(fused) for head in self.move_heads], dim=1)
        return (
            move_logits,
            self.item_head(fused),
            self.ability_head(fused),
            self.tera_head(fused),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        turn_mask: torch.Tensor,
        player_species_ids: torch.Tensor,
        opponent_species_ids: torch.Tensor,
    ) -> ActiveBeliefOutput:
        final_turn = self.encode_final_turn(token_ids, turn_mask)
        matchup = self.matchup_encoder(player_species_ids, opponent_species_ids)
        fused = final_turn + matchup
        move_logits, item_logits, ability_logits, tera_logits = self.belief_logits(
            fused
        )
        policy_input = self._policy_input(
            fused, move_logits, item_logits, ability_logits, tera_logits
        )
        return ActiveBeliefOutput(
            action_logits=self.action_head(policy_input),
            move_logits=move_logits,
            item_logits=item_logits,
            ability_logits=ability_logits,
            tera_logits=tera_logits,
        )

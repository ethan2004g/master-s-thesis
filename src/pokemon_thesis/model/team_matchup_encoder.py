from __future__ import annotations

import torch
import torch.nn as nn


class TeamMatchupEncoder(nn.Module):
    """Encode both team compositions into one matchup vector."""

    def __init__(self, n_species: int, d_model: int) -> None:
        super().__init__()
        self.species_embed = nn.Embedding(n_species, d_model, padding_idx=0)
        self.proj = nn.Linear(d_model, d_model)

    def _pool_team(self, species_ids: torch.Tensor) -> torch.Tensor:
        emb = self.species_embed(species_ids)
        mask = (species_ids != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (emb * mask).sum(dim=1) / denom

    def forward(
        self,
        player_species_ids: torch.Tensor,
        opponent_species_ids: torch.Tensor,
    ) -> torch.Tensor:
        player = self._pool_team(player_species_ids)
        opponent = self._pool_team(opponent_species_ids)
        return self.proj(player + opponent)

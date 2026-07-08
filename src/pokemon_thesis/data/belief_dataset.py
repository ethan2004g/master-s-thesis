from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from pokemon_thesis.data.action_vocab import ActionVocab
from pokemon_thesis.data.belief_labels import SpeciesBeliefVocab, opponent_species_labels
from pokemon_thesis.data.encoded_dataset import (
    build_context_window,
    group_trajectories,
    load_encoded_records,
)


class BeliefBattleDataset(Dataset):
    """Causal context windows with opponent species belief targets."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        action_vocab: ActionVocab,
        species_vocab: SpeciesBeliefVocab,
        context_turns: int = 8,
        sample_filter: Optional[Any] = None,
    ) -> None:
        if context_turns < 1:
            raise ValueError("context_turns must be at least 1")

        self.context_turns = context_turns
        self.action_vocab = action_vocab
        self.species_vocab = species_vocab
        self.samples: list[dict[str, Any]] = []

        grouped = group_trajectories(records)
        for trajectory in grouped.values():
            frames: list[list[int]] = []
            for record in trajectory:
                if sample_filter and not sample_filter(record):
                    continue
                frames.append(record["token_ids"])
                context, mask = build_context_window(frames, context_turns)
                action = record.get("action") or {}
                species_labels, species_mask = opponent_species_labels(
                    record, species_vocab
                )
                self.samples.append(
                    {
                        "token_ids": context,
                        "turn_mask": mask,
                        "action_id": action_vocab.encode(action),
                        "species_labels": species_labels,
                        "species_mask": species_mask,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "token_ids": torch.tensor(sample["token_ids"], dtype=torch.long),
            "turn_mask": torch.tensor(sample["turn_mask"], dtype=torch.float),
            "action_id": torch.tensor(sample["action_id"], dtype=torch.long),
            "species_labels": torch.tensor(sample["species_labels"], dtype=torch.long),
            "species_mask": torch.tensor(sample["species_mask"], dtype=torch.float),
        }

    @classmethod
    def from_paths(
        cls,
        paths: list[Any],
        action_vocab: ActionVocab,
        species_vocab: SpeciesBeliefVocab,
        context_turns: int = 8,
        battle_tags: Optional[set[str]] = None,
    ) -> BeliefBattleDataset:
        records = load_encoded_records(paths)
        sample_filter = None
        if battle_tags is not None:
            sample_filter = lambda row: row.get("battle_tag") in battle_tags
        return cls(records, action_vocab, species_vocab, context_turns, sample_filter)

    def count_supervised_species_slots(self) -> int:
        total = 0.0
        for sample in self.samples:
            total += sum(sample["species_mask"])
        return int(total)

    @staticmethod
    def split_battle_tags(
        records: list[dict[str, Any]],
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[set[str], set[str]]:
        from pokemon_thesis.data.encoded_dataset import EncodedBattleDataset

        return EncodedBattleDataset.split_battle_tags(records, val_fraction, seed)

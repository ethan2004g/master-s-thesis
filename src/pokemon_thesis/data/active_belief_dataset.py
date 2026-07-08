from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from pokemon_thesis.data.action_vocab import ActionVocab
from pokemon_thesis.data.active_belief_labels import (
    BeliefLabelVocab,
    active_belief_tensors,
    matchup_species_ids,
)
from pokemon_thesis.data.encoded_dataset import (
    build_context_window,
    group_trajectories,
    load_encoded_records,
)


class ActiveBeliefDataset(Dataset):
    """Human replay windows with matchup ids and active-opponent belief targets."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        action_vocab: ActionVocab,
        belief_vocabs: dict[str, BeliefLabelVocab],
        context_turns: int = 8,
        sample_filter: Optional[Any] = None,
    ) -> None:
        if context_turns < 1:
            raise ValueError("context_turns must be at least 1")

        self.context_turns = context_turns
        self.action_vocab = action_vocab
        self.belief_vocabs = belief_vocabs
        self.samples: list[dict[str, Any]] = []

        grouped = group_trajectories(records)
        for trajectory in grouped.values():
            frames: list[list[int]] = []
            for record in trajectory:
                if sample_filter and not sample_filter(record):
                    continue
                frames.append(record["token_ids"])
                context, mask = build_context_window(frames, context_turns)
                belief = active_belief_tensors(record, belief_vocabs)
                player_ids, opponent_ids = matchup_species_ids(
                    record, belief_vocabs["species"]
                )
                self.samples.append(
                    {
                        "token_ids": context,
                        "turn_mask": mask,
                        "action_id": action_vocab.encode(record.get("action") or {}),
                        "player_species_ids": player_ids,
                        "opponent_species_ids": opponent_ids,
                        **belief,
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
            "player_species_ids": torch.tensor(
                sample["player_species_ids"], dtype=torch.long
            ),
            "opponent_species_ids": torch.tensor(
                sample["opponent_species_ids"], dtype=torch.long
            ),
            "move_labels": torch.tensor(sample["move_labels"], dtype=torch.long),
            "move_mask": torch.tensor(sample["move_mask"], dtype=torch.float),
            "item_label": torch.tensor(sample["item_label"], dtype=torch.long),
            "item_mask": torch.tensor(sample["item_mask"], dtype=torch.float),
            "ability_label": torch.tensor(sample["ability_label"], dtype=torch.long),
            "ability_mask": torch.tensor(sample["ability_mask"], dtype=torch.float),
            "tera_label": torch.tensor(sample["tera_label"], dtype=torch.long),
            "tera_mask": torch.tensor(sample["tera_mask"], dtype=torch.float),
        }

    @classmethod
    def from_paths(
        cls,
        paths: list[Any],
        action_vocab: ActionVocab,
        belief_vocabs: dict[str, BeliefLabelVocab],
        context_turns: int = 8,
        battle_tags: Optional[set[str]] = None,
    ) -> ActiveBeliefDataset:
        records = load_encoded_records(paths)
        sample_filter = None
        if battle_tags is not None:
            sample_filter = lambda row: row.get("battle_tag") in battle_tags
        return cls(records, action_vocab, belief_vocabs, context_turns, sample_filter)

    def supervised_counts(self) -> dict[str, int]:
        totals = {
            "move_slots": 0,
            "item": 0,
            "ability": 0,
            "tera": 0,
        }
        for sample in self.samples:
            totals["move_slots"] += int(sum(sample["move_mask"]))
            totals["item"] += int(sample["item_mask"])
            totals["ability"] += int(sample["ability_mask"])
            totals["tera"] += int(sample["tera_mask"])
        return totals

    @staticmethod
    def split_battle_tags(
        records: list[dict[str, Any]],
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[set[str], set[str]]:
        from pokemon_thesis.data.encoded_dataset import EncodedBattleDataset

        return EncodedBattleDataset.split_battle_tags(records, val_fraction, seed)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset

from pokemon_thesis.data.action_vocab import ActionVocab
from pokemon_thesis.tokenizer.turn_tokenizer import TURN_TOKEN_COUNT

TrajectoryKey = tuple[str, str]


def load_encoded_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_trajectories(
    records: list[dict[str, Any]],
) -> dict[TrajectoryKey, list[dict[str, Any]]]:
    grouped: dict[TrajectoryKey, list[dict[str, Any]]] = {}
    for record in records:
        key = (record["battle_tag"], record.get("player", "p1"))
        grouped.setdefault(key, []).append(record)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda row: row.get("turn", 0))
    return grouped


def build_context_window(
    token_frames: list[list[int]],
    context_turns: int,
) -> tuple[list[list[int]], list[float]]:
    """Left-pad so the most recent frame is always the last slot."""
    history = token_frames[-context_turns:]
    pad_count = context_turns - len(history)
    padded = [[0] * TURN_TOKEN_COUNT for _ in range(pad_count)] + history
    mask = [0.0] * pad_count + [1.0] * len(history)
    return padded, mask


class EncodedBattleDataset(Dataset):
    """
    Each sample is a causal context of up to N prior turns plus the current turn.

    token_ids shape is (context_turns, 13).
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        action_vocab: ActionVocab,
        context_turns: int = 8,
        sample_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
    ) -> None:
        if context_turns < 1:
            raise ValueError("context_turns must be at least 1")

        self.context_turns = context_turns
        self.action_vocab = action_vocab
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
                self.samples.append(
                    {
                        "token_ids": context,
                        "turn_mask": mask,
                        "action_id": action_vocab.encode(action),
                        "battle_tag": record.get("battle_tag"),
                        "turn": record.get("turn"),
                        "player": record.get("player"),
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
        }

    @classmethod
    def from_paths(
        cls,
        paths: list[Path],
        action_vocab: ActionVocab,
        context_turns: int = 8,
        battle_tags: Optional[set[str]] = None,
    ) -> EncodedBattleDataset:
        records = load_encoded_records(paths)
        sample_filter = None
        if battle_tags is not None:
            sample_filter = lambda row: row.get("battle_tag") in battle_tags
        return cls(records, action_vocab, context_turns, sample_filter)

    @staticmethod
    def split_battle_tags(
        records: list[dict[str, Any]],
        val_fraction: float = 0.2,
        seed: int = 0,
    ) -> tuple[set[str], set[str]]:
        tags = sorted({record["battle_tag"] for record in records})
        if not tags:
            return set(), set()
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(tags), generator=rng).tolist()
        shuffled = [tags[i] for i in perm]
        val_count = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 1
        val_tags = set(shuffled[:val_count])
        train_tags = set(shuffled[val_count:]) or val_tags
        return train_tags, val_tags

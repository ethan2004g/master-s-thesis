from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

UNKNOWN = "<unknown>"


class BeliefLabelVocab:
    """Generic string-to-id vocabulary for belief targets."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if UNKNOWN not in token_to_id:
            raise ValueError(f"vocab must contain {UNKNOWN!r}")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, token: Optional[str]) -> int:
        if not token or token == UNKNOWN:
            return self.token_to_id[UNKNOWN]
        return self.token_to_id.setdefault(token, len(self.token_to_id))

    def decode(self, label_id: int) -> str:
        return self.id_to_token[label_id]

    @classmethod
    def empty(cls) -> BeliefLabelVocab:
        return cls({UNKNOWN: 0})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> BeliefLabelVocab:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["token_to_id"])


def build_active_belief_vocabs(records: list[dict[str, Any]]) -> dict[str, BeliefLabelVocab]:
    vocabs = {
        "move": BeliefLabelVocab.empty(),
        "item": BeliefLabelVocab.empty(),
        "ability": BeliefLabelVocab.empty(),
        "tera": BeliefLabelVocab.empty(),
        "species": BeliefLabelVocab.empty(),
    }
    for record in records:
        belief = record.get("belief") or {}
        for move in belief.get("move_labels") or []:
            if move:
                vocabs["move"].encode(move)
        for field in ("item_label", "ability_label", "tera_label"):
            value = belief.get(field)
            if value:
                key = field.replace("_label", "")
                vocabs[key].encode(value)
        matchup = record.get("team_matchup") or {}
        for species in (matchup.get("player_species") or []) + (
            matchup.get("opponent_species") or []
        ):
            if species:
                vocabs["species"].encode(species)
    return vocabs


def active_belief_tensors(
    record: dict[str, Any],
    vocabs: dict[str, BeliefLabelVocab],
    move_slots: int = 4,
) -> dict[str, Any]:
    belief = record.get("belief") or {}
    move_labels = [-1] * move_slots
    move_mask = [0.0] * move_slots
    for slot in range(move_slots):
        raw_labels = belief.get("move_labels") or []
        raw_masks = belief.get("move_mask") or []
        if slot >= len(raw_labels) or slot >= len(raw_masks) or not raw_masks[slot]:
            continue
        label = raw_labels[slot]
        if label:
            move_labels[slot] = vocabs["move"].encode(label)
            move_mask[slot] = 1.0

    item_label = -1
    item_mask = 0.0
    if belief.get("item_mask") and belief.get("item_label"):
        item_label = vocabs["item"].encode(belief["item_label"])
        item_mask = 1.0

    ability_label = -1
    ability_mask = 0.0
    if belief.get("ability_mask") and belief.get("ability_label"):
        ability_label = vocabs["ability"].encode(belief["ability_label"])
        ability_mask = 1.0

    tera_label = -1
    tera_mask = 0.0
    if belief.get("tera_mask") and belief.get("tera_label"):
        tera_label = vocabs["tera"].encode(belief["tera_label"])
        tera_mask = 1.0

    return {
        "move_labels": move_labels,
        "move_mask": move_mask,
        "item_label": item_label,
        "item_mask": item_mask,
        "ability_label": ability_label,
        "ability_mask": ability_mask,
        "tera_label": tera_label,
        "tera_mask": tera_mask,
    }


def matchup_species_ids(
    record: dict[str, Any],
    species_vocab: BeliefLabelVocab,
    party_size: int = 6,
) -> tuple[list[int], list[int]]:
    matchup = record.get("team_matchup") or {}
    player = [0] * party_size
    opponent = [0] * party_size
    for slot in range(party_size):
        player_raw = (matchup.get("player_species") or [None] * party_size)[slot]
        opponent_raw = (matchup.get("opponent_species") or [None] * party_size)[slot]
        player[slot] = species_vocab.encode(player_raw) if player_raw else 0
        opponent[slot] = species_vocab.encode(opponent_raw) if opponent_raw else 0
    return player, opponent

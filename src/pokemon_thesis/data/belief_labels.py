from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pokemon_thesis.tokenizer.turn_tokenizer import PARTY_SIZE

MASK_LABEL = -1
UNKNOWN_SPECIES = "<unknown>"


class SpeciesBeliefVocab:
    """Maps opponent species names to integer belief labels."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        if UNKNOWN_SPECIES not in token_to_id:
            raise ValueError(f"vocab must contain {UNKNOWN_SPECIES!r}")
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, species: Optional[str]) -> int:
        if not species or species == UNKNOWN_SPECIES:
            return self.token_to_id[UNKNOWN_SPECIES]
        return self.token_to_id.setdefault(species, len(self.token_to_id))

    def decode(self, label_id: int) -> str:
        return self.id_to_token[label_id]

    @classmethod
    def build_from_records(cls, records: list[dict[str, Any]]) -> SpeciesBeliefVocab:
        vocab = cls({UNKNOWN_SPECIES: 0})
        for record in records:
            belief = record.get("belief") or {}
            for species in belief.get("opponent_species") or []:
                if species and species != UNKNOWN_SPECIES:
                    vocab.encode(species)
        return vocab

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"token_to_id": self.token_to_id}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> SpeciesBeliefVocab:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["token_to_id"])


def opponent_species_labels(
    record: dict[str, Any],
    species_vocab: SpeciesBeliefVocab,
) -> tuple[list[int], list[float]]:
    """
    Build per-slot species labels and mask from a record.

    Returns
        labels  length-6 list with MASK_LABEL for masked slots
        mask    length-6 list with 1.0 for supervised slots, 0.0 otherwise
    """
    belief = record.get("belief") or {}
    raw_species = belief.get("opponent_species")
    raw_mask = belief.get("opponent_species_mask")

    labels = [MASK_LABEL] * PARTY_SIZE
    mask = [0.0] * PARTY_SIZE

    if raw_species is None:
        return labels, mask

    for slot in range(PARTY_SIZE):
        if raw_mask is not None and not raw_mask[slot]:
            continue
        species = raw_species[slot] if slot < len(raw_species) else None
        if not species or species == UNKNOWN_SPECIES:
            continue
        labels[slot] = species_vocab.encode(species)
        mask[slot] = 1.0

    return labels, mask


def attach_belief_from_ground_truth(record: dict[str, Any]) -> dict[str, Any]:
    """
    Attach belief targets when ground-truth opponent species are available.

    Visible species in the first-person opponent_team view are masked out.
    Ground truth comes from belief.ground_truth_opponent_team or
    ground_truth_opponent_team on the record.
    """
    view_team = record.get("opponent_team") or []
    gt_team = None
    belief = record.get("belief") or {}
    gt_team = belief.get("ground_truth_opponent_team") or record.get(
        "ground_truth_opponent_team"
    )
    if gt_team is None:
        return record

    visible: set[str] = set()
    for mon in view_team:
        if mon and mon.get("species"):
            visible.add(mon["species"])

    opponent_species: list[Optional[str]] = []
    opponent_species_mask: list[bool] = []
    for slot in range(PARTY_SIZE):
        mon = gt_team[slot] if slot < len(gt_team) else None
        species = mon.get("species") if mon else None
        opponent_species.append(species)
        opponent_species_mask.append(bool(species and species not in visible))

    updated = dict(record)
    updated["belief"] = {
        **belief,
        "opponent_species": opponent_species,
        "opponent_species_mask": opponent_species_mask,
    }
    return updated

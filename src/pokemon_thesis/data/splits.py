from __future__ import annotations

from typing import Any, Optional

import torch


def _species_from_party_token(token: str) -> Optional[str]:
    if not token or token in {"<pad>", "pad"}:
        return None
    if token.startswith("party|"):
        parts = token.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return None


def opponent_team_key(record: dict[str, Any]) -> Optional[str]:
    """Canonical key from sorted opponent species in ground truth or view."""
    matchup = record.get("team_matchup") or {}
    matchup_species = matchup.get("opponent_species")
    if matchup_species:
        species = sorted({s for s in matchup_species if s})
        if species:
            return "|".join(species)

    token_strings = record.get("token_strings")
    if token_strings and len(token_strings) >= 13:
        species = sorted(
            {
                s
                for token in token_strings[7:13]
                if (s := _species_from_party_token(token))
            }
        )
        if species:
            return "|".join(species)

    belief = record.get("belief") or {}
    gt_team = belief.get("ground_truth_opponent_team") or record.get(
        "ground_truth_opponent_team"
    )
    team = gt_team or record.get("opponent_team") or []
    species = sorted(
        {
            mon.get("species")
            for mon in team
            if mon and mon.get("species")
        }
    )
    if not species:
        return None
    return "|".join(species)


def collect_team_keys(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Map each team key to battle tags that use it."""
    key_to_battles: dict[str, set[str]] = {}
    for record in records:
        key = opponent_team_key(record)
        tag = record.get("battle_tag")
        if key and tag:
            key_to_battles.setdefault(key, set()).add(tag)
    return key_to_battles


def held_out_team_split(
    records: list[dict[str, Any]],
    holdout_fraction: float = 0.2,
    seed: int = 0,
    min_battles_per_key: int = 1,
) -> tuple[set[str], set[str]]:
    """
    Hold out entire opponent team keys from training.

    Returns train battle tags and test battle tags.
    """
    key_to_battles = collect_team_keys(records)
    eligible = [
        key for key, tags in key_to_battles.items() if len(tags) >= min_battles_per_key
    ]
    if not eligible:
        all_tags = {record["battle_tag"] for record in records}
        return all_tags, set()

    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(eligible), generator=rng).tolist()
    shuffled = [eligible[i] for i in perm]
    holdout_count = max(1, int(len(shuffled) * holdout_fraction))
    holdout_keys = set(shuffled[:holdout_count])

    test_tags: set[str] = set()
    train_tags: set[str] = set()
    for key, tags in key_to_battles.items():
        if key in holdout_keys:
            test_tags.update(tags)
        else:
            train_tags.update(tags)

    train_tags -= test_tags
    return train_tags, test_tags

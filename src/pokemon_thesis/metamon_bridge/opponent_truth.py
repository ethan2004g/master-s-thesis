from __future__ import annotations

from typing import Any, Optional

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.interface import UniversalPokemon, UniversalState, consistent_pokemon_order

UNKNOWN_ITEM = "unknownitem"
UNKNOWN_ABILITY = "unknownability"
UNKNOWN_TERA = "notype"
NO_MOVE = "nomove"
MOVE_SLOTS = 4
PARTY_SIZE = 6


def _species_name(mon: Optional[UniversalPokemon]) -> Optional[str]:
    if mon is None:
        return None
    return mon.base_species or mon.name


def _visible_move_names(mon: UniversalPokemon) -> set[str]:
    names: set[str] = set()
    for move in mon.moves:
        if move.name and move.name != NO_MOVE:
            names.add(move.name)
    return names


def _slot_moves(move_names: set[str]) -> list[str]:
    ordered = sorted(move_names)[:MOVE_SLOTS]
    while len(ordered) < MOVE_SLOTS:
        ordered.append(NO_MOVE)
    return ordered


def accumulate_opponent_set_truth(
    states: list[UniversalState],
) -> dict[str, dict[str, Any]]:
    """Merge opponent set knowledge seen whenever a species is active."""
    truth: dict[str, dict[str, Any]] = {}
    for state in states:
        mon = state.opponent_active_pokemon
        species = _species_name(mon)
        if not species:
            continue
        entry = truth.setdefault(
            species,
            {"moves": set(), "item": None, "ability": None, "tera": None},
        )
        entry["moves"].update(_visible_move_names(mon))
        if mon.item not in {None, UNKNOWN_ITEM, "noitem"}:
            entry["item"] = mon.item
        if mon.ability not in {None, UNKNOWN_ABILITY, "noability"}:
            entry["ability"] = mon.ability
        if mon.tera_type and mon.tera_type.strip() not in {UNKNOWN_TERA, ""}:
            entry["tera"] = mon.tera_type.strip()
    return truth


def full_player_roster(states: list[UniversalState]) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    seen: set[str] = set()
    for state in states:
        candidates = [state.player_active_pokemon, *consistent_pokemon_order(state.available_switches)]
        for mon in candidates:
            species = _species_name(mon)
            if not species or species in seen:
                continue
            roster.append({"species": species})
            seen.add(species)
    while len(roster) < PARTY_SIZE:
        roster.append({"species": None})
    return roster[:PARTY_SIZE]


def full_opponent_roster(states: list[UniversalState]) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    seen: set[str] = set()

    for state in states:
        if state.opponent_active_pokemon:
            species = _species_name(state.opponent_active_pokemon)
            if species and species not in seen:
                roster.append({"species": species})
                seen.add(species)
        for preview_name in state.opponent_teampreview:
            species = pokemon_name(preview_name)
            if species not in seen:
                roster.append({"species": species})
                seen.add(species)

    while len(roster) < PARTY_SIZE:
        roster.append({"species": None})
    return roster[:PARTY_SIZE]


def team_matchup_from_states(states: list[UniversalState]) -> dict[str, list[Optional[str]]]:
    player = [slot.get("species") for slot in full_player_roster(states)]
    opponent = [slot.get("species") for slot in full_opponent_roster(states)]
    return {"player_species": player, "opponent_species": opponent}


def active_opponent_belief_targets(
    state: UniversalState,
    truth_by_species: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build belief labels for the currently active opponent Pokemon."""
    mon = state.opponent_active_pokemon
    species = _species_name(mon)
    if not species:
        return {}

    gt = truth_by_species.get(species, {})
    visible_moves = _visible_move_names(mon)
    gt_moves = set(gt.get("moves") or [])
    gt_move_slots = _slot_moves(gt_moves)
    visible_move_slots = _slot_moves(visible_moves)

    move_labels: list[Optional[str]] = []
    move_mask: list[bool] = []
    for slot in range(MOVE_SLOTS):
        visible = visible_move_slots[slot]
        target = gt_move_slots[slot]
        if target == NO_MOVE:
            move_labels.append(None)
            move_mask.append(False)
        elif visible != NO_MOVE:
            move_labels.append(None)
            move_mask.append(False)
        else:
            move_labels.append(target)
            move_mask.append(True)

    item_label = gt.get("item")
    item_mask = bool(item_label and mon.item == UNKNOWN_ITEM)

    ability_label = gt.get("ability")
    ability_mask = bool(ability_label and mon.ability == UNKNOWN_ABILITY)

    tera_label = gt.get("tera")
    tera_mask = bool(
        tera_label
        and (not mon.tera_type or mon.tera_type.strip() in {UNKNOWN_TERA, ""})
    )

    return {
        "active_species": species,
        "move_labels": move_labels,
        "move_mask": move_mask,
        "item_label": item_label if item_mask else None,
        "item_mask": item_mask,
        "ability_label": ability_label if ability_mask else None,
        "ability_mask": ability_mask,
        "tera_label": tera_label if tera_mask else None,
        "tera_mask": tera_mask,
    }


def attach_belief_and_matchup(
    record: dict[str, Any],
    state: UniversalState,
    states: list[UniversalState],
) -> dict[str, Any]:
    truth = accumulate_opponent_set_truth(states)
    active = active_opponent_belief_targets(state, truth)
    matchup = team_matchup_from_states(states)

    updated = dict(record)
    updated["ground_truth_opponent_team"] = full_opponent_roster(states)
    updated["team_matchup"] = matchup
    updated["belief"] = {
        **(record.get("belief") or {}),
        **active,
        "ground_truth_opponent_sets": {
            species: {
                "moves": sorted(data.get("moves") or []),
                "item": data.get("item"),
                "ability": data.get("ability"),
                "tera": data.get("tera"),
            }
            for species, data in truth.items()
        },
    }
    return updated

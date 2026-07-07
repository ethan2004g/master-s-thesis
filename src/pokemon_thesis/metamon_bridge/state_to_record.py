from __future__ import annotations

from typing import Any, Optional

from metamon.backend.replay_parser.str_parsing import move_name, pokemon_name
from metamon.interface import UniversalPokemon, UniversalState, consistent_move_order, consistent_pokemon_order


def _normalize_status(status: str | None) -> Optional[str]:
    if not status or status in {"nostatus", "noeffect"}:
        return None
    cleaned = status.upper()
    if cleaned == "FNT":
        return "FNT"
    return cleaned


def _pokemon_snapshot(mon: UniversalPokemon, active: bool) -> dict[str, Any]:
    status = _normalize_status(mon.status)
    hp = float(mon.hp_pct)
    fainted = hp <= 0.0 or status == "FNT"
    species = mon.base_species or mon.name
    return {
        "species": species,
        "hp_fraction": round(hp, 4),
        "status": status,
        "fainted": fainted,
        "active": active,
    }


def _player_team(state: UniversalState) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    active = _pokemon_snapshot(state.player_active_pokemon, active=True)
    slots.append(active)
    seen = {active["species"]}

    for switch in consistent_pokemon_order(state.available_switches):
        snap = _pokemon_snapshot(switch, active=False)
        if snap["species"] in seen:
            continue
        slots.append(snap)
        seen.add(snap["species"])
    return slots


def _opponent_team(state: UniversalState) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()

    if state.opponent_active_pokemon:
        active = _pokemon_snapshot(state.opponent_active_pokemon, active=True)
        slots.append(active)
        seen.add(active["species"])

    for preview_name in state.opponent_teampreview:
        species = pokemon_name(preview_name)
        if species in seen:
            continue
        slots.append(
            {
                "species": species,
                "hp_fraction": 1.0,
                "status": None,
                "fainted": False,
                "active": False,
            }
        )
        seen.add(species)
    return slots


def _field_dict(value: str, empty_values: set[str]) -> dict[str, int]:
    if not value or value in empty_values:
        return {}
    return {value: 1}


def action_idx_to_dict(state: UniversalState, action_idx: int) -> dict[str, Any]:
    if action_idx < 0:
        return {"kind": "other", "message": "missing"}

    moves = consistent_move_order(state.player_active_pokemon.moves)
    switches = consistent_pokemon_order(state.available_switches)
    idx = action_idx

    if idx >= 9:
        idx -= 9

    if idx <= 3 and idx < len(moves):
        return {"kind": "move", "move_id": move_name(moves[idx].name)}

    if 4 <= idx <= 8:
        switch_idx = idx - 4
        if switch_idx < len(switches):
            target = switches[switch_idx]
            species = target.base_species or target.name
            return {"kind": "switch", "species": species}

    return {"kind": "other", "message": f"idx_{action_idx}"}


def full_opponent_roster(states: list[UniversalState]) -> list[dict[str, Any]]:
    """Collect up to six opponent species seen across a battle trajectory."""
    roster: list[dict[str, Any]] = []
    seen: set[str] = set()

    for state in states:
        if state.opponent_active_pokemon:
            species = (
                state.opponent_active_pokemon.base_species
                or state.opponent_active_pokemon.name
            )
            if species not in seen:
                roster.append({"species": species})
                seen.add(species)

        for preview_name in state.opponent_teampreview:
            species = pokemon_name(preview_name)
            if species not in seen:
                roster.append({"species": species})
                seen.add(species)

    while len(roster) < 6:
        roster.append({"species": None})
    return roster[:6]


def universal_state_to_record(
    state: UniversalState,
    turn: int,
    battle_tag: str,
    action_idx: int,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "battle_tag": battle_tag,
        "player": "p1",
        "weather": _field_dict(state.weather, {"noweather"}),
        "fields": _field_dict(state.battle_field, {"nofield"}),
        "side_conditions": _field_dict(state.player_conditions, {"noconditions"}),
        "opponent_side_conditions": _field_dict(
            state.opponent_conditions, {"noconditions"}
        ),
        "team": _player_team(state),
        "opponent_team": _opponent_team(state),
        "action": action_idx_to_dict(state, action_idx),
    }

from __future__ import annotations

from typing import Any, Optional

from poke_env.battle import AbstractBattle, Battle, Move, Pokemon
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder


def enum_names(mapping: dict) -> dict[str, int]:
    return {key.name: value for key, value in mapping.items()}


def pokemon_snapshot(mon: Optional[Pokemon]) -> Optional[dict[str, Any]]:
    if mon is None:
        return None
    return {
        "species": mon.species,
        "hp_fraction": round(mon.current_hp_fraction, 4),
        "status": mon.status.name if mon.status is not None else None,
        "fainted": mon.fainted,
        "active": mon.active,
    }


def party_snapshot(team: dict[str, Pokemon]) -> list[dict[str, Any]]:
    return [pokemon_snapshot(mon) for mon in team.values()]


def describe_order(order: BattleOrder) -> dict[str, Any]:
    if isinstance(order, SingleBattleOrder):
        chosen = order.order
        if isinstance(chosen, Move):
            return {"kind": "move", "move_id": chosen.id}
        if isinstance(chosen, Pokemon):
            return {"kind": "switch", "species": chosen.species}
    return {"kind": "other", "message": order.message}


def battle_snapshot(battle: AbstractBattle) -> dict[str, Any]:
    weather = enum_names(battle.weather)
    fields = enum_names(battle.fields)
    side_conditions = enum_names(battle.side_conditions)
    opponent_side_conditions = enum_names(battle.opponent_side_conditions)

    active = None
    opponent_active = None
    if isinstance(battle, Battle):
        active = pokemon_snapshot(battle.active_pokemon)
        opponent_active = pokemon_snapshot(battle.opponent_active_pokemon)

    return {
        "turn": battle.turn,
        "battle_tag": battle.battle_tag,
        "weather": weather,
        "fields": fields,
        "side_conditions": side_conditions,
        "opponent_side_conditions": opponent_side_conditions,
        "active_pokemon": active,
        "opponent_active_pokemon": opponent_active,
        "team": party_snapshot(battle.team),
        "opponent_team": party_snapshot(battle.opponent_team),
    }

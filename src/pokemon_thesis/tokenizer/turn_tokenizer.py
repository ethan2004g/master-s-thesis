from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pokemon_thesis.tokenizer.vocab import PAD, TokenVocab

PARTY_SIZE = 6
TURN_TOKEN_COUNT = 13


@dataclass(frozen=True)
class TurnTokens:
    """Exactly 13 token strings and their integer ids for one turn."""

    token_strings: tuple[str, ...]
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.token_strings) != TURN_TOKEN_COUNT:
            raise ValueError(
                f"Expected {TURN_TOKEN_COUNT} tokens, got {len(self.token_strings)}"
            )
        if len(self.token_ids) != TURN_TOKEN_COUNT:
            raise ValueError(
                f"Expected {TURN_TOKEN_COUNT} token ids, got {len(self.token_ids)}"
            )


class TurnTokenizer:
    """
    Encodes a battle turn as 13 discrete tokens.

    Layout
      0       field (weather, terrain, hazards)
      1-6     player party slots
      7-12    opponent party slots
    """

    def __init__(self, vocab: Optional[TokenVocab] = None) -> None:
        self.vocab = vocab or TokenVocab()

    def encode_record(self, record: dict[str, Any]) -> TurnTokens:
        strings = self.token_strings_from_record(record)
        ids = tuple(self.vocab.encode(token) for token in strings)
        return TurnTokens(token_strings=strings, token_ids=ids)

    def token_strings_from_record(self, record: dict[str, Any]) -> tuple[str, ...]:
        field = self.field_token(record)
        party = self.party_tokens(record.get("team") or [])
        opponent = self.party_tokens(record.get("opponent_team") or [])
        return (field, *party, *opponent)

    @staticmethod
    def field_token(record: dict[str, Any]) -> str:
        weather = TurnTokenizer._sorted_effect_keys(record.get("weather") or {})
        fields = TurnTokenizer._sorted_effect_keys(record.get("fields") or {})
        hazards = TurnTokenizer._sorted_effect_keys(record.get("side_conditions") or {})
        opp_hazards = TurnTokenizer._sorted_effect_keys(
            record.get("opponent_side_conditions") or {}
        )

        if not weather and not fields and not hazards and not opp_hazards:
            return "field|clear"

        parts = ["field"]
        if weather:
            parts.append("w=" + ",".join(weather))
        if fields:
            parts.append("t=" + ",".join(fields))
        if hazards:
            parts.append("h=" + ",".join(hazards))
        if opp_hazards:
            parts.append("oh=" + ",".join(opp_hazards))
        return "|".join(parts)

    @staticmethod
    def _sorted_effect_keys(effects: dict[str, Any]) -> list[str]:
        return sorted(effects.keys())

    @staticmethod
    def party_tokens(team: list[dict[str, Any]]) -> tuple[str, ...]:
        slots = list(team[:PARTY_SIZE])
        while len(slots) < PARTY_SIZE:
            slots.append(None)
        return tuple(TurnTokenizer.pokemon_token(mon) for mon in slots[:PARTY_SIZE])

    @staticmethod
    def pokemon_token(mon: Optional[dict[str, Any]]) -> str:
        if mon is None:
            return PAD

        species = mon.get("species") or "unknown"
        hp = TurnTokenizer.hp_bucket(mon.get("hp_fraction", 0.0))
        status = mon.get("status") or "none"
        active = int(bool(mon.get("active")))
        fainted = int(bool(mon.get("fainted")))

        return (
            f"party|{species}|hp={hp}|status={status}"
            f"|active={active}|faint={fainted}"
        )

    @staticmethod
    def hp_bucket(fraction: float) -> str:
        if fraction <= 0:
            return "zero"
        if fraction >= 1.0:
            return "full"
        if fraction <= 0.25:
            return "low"
        if fraction <= 0.50:
            return "midlow"
        if fraction <= 0.75:
            return "midhigh"
        return "high"

    def decode_turn(self, token_ids: tuple[int, ...]) -> tuple[str, ...]:
        return tuple(self.vocab.decode(token_id) for token_id in token_ids)

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from poke_env.battle import AbstractBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from pokemon_thesis.battle_snapshot import battle_snapshot
from pokemon_thesis.data.encoded_dataset import build_context_window
from pokemon_thesis.inference.checkpoint import BCCheckpoint, load_bc_checkpoint
from pokemon_thesis.tokenizer.turn_tokenizer import TurnTokenizer


class BCPlayer(Player):
    """Play battles using a trained behavioral cloning checkpoint."""

    def __init__(
        self,
        checkpoint: BCCheckpoint | None = None,
        checkpoint_dir: Path | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if checkpoint is None:
            if checkpoint_dir is None:
                checkpoint_dir = Path("checkpoints")
            checkpoint = load_bc_checkpoint(Path(checkpoint_dir))
        self._checkpoint = checkpoint
        self._tokenizer = TurnTokenizer(vocab=checkpoint.token_vocab)
        self._histories: dict[str, list[list[int]]] = {}

    def teampreview(self, battle: AbstractBattle) -> str:
        return self.random_teampreview(battle)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        self._histories.pop(battle.battle_tag, None)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        record = battle_snapshot(battle)
        token_strings = self._tokenizer.token_strings_from_record(record)
        frame = [self._checkpoint.token_vocab.lookup(token) for token in token_strings]
        history = self._histories.setdefault(battle.battle_tag, [])
        history.append(frame)

        context, mask = build_context_window(history, self._checkpoint.context_turns)
        token_ids = torch.tensor([context], dtype=torch.long, device=self._checkpoint.device)
        turn_mask = torch.tensor([mask], dtype=torch.float, device=self._checkpoint.device)

        with torch.no_grad():
            logits = self._checkpoint.model(token_ids, turn_mask)

        action_id = int(logits.argmax(dim=-1).item())
        action_key = self._checkpoint.action_vocab.decode(action_id)
        order = self._action_to_order(action_key, battle)
        if order is not None:
            return order
        return self.choose_random_move(battle)

    @staticmethod
    def _action_to_order(action_key: str, battle: AbstractBattle) -> BattleOrder | None:
        if action_key.startswith("move|"):
            move_id = action_key.split("|", 1)[1]
            for move in battle.available_moves:
                if move.id == move_id:
                    return Player.create_order(move)
        if action_key.startswith("switch|"):
            species = action_key.split("|", 1)[1]
            for pokemon in battle.available_switches:
                if pokemon.species == species:
                    return Player.create_order(pokemon)
        return None

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


UNK_ACTION = "<unk_action>"


class ActionVocab:
    """Maps logged actions to class ids for behavioral cloning."""

    def __init__(self) -> None:
        self.action_to_id: dict[str, int] = {UNK_ACTION: 0}
        self.id_to_action: dict[int, str] = {0: UNK_ACTION}

    def __len__(self) -> int:
        return len(self.action_to_id)

    @staticmethod
    def action_key(action: dict[str, Any]) -> str:
        kind = action.get("kind", "other")
        if kind == "move":
            move_id = action.get("move_id") or "unknown"
            return f"move|{move_id}"
        if kind == "switch":
            species = action.get("species") or "unknown"
            return f"switch|{species}"
        message = action.get("message") or "unknown"
        return f"other|{message}"

    def add(self, action: dict[str, Any]) -> int:
        key = self.action_key(action)
        if key in self.action_to_id:
            return self.action_to_id[key]
        next_id = len(self.action_to_id)
        self.action_to_id[key] = next_id
        self.id_to_action[next_id] = key
        return next_id

    def encode(self, action: dict[str, Any]) -> int:
        return self.add(action)

    def decode(self, action_id: int) -> str:
        return self.id_to_action.get(action_id, UNK_ACTION)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"action_to_id": self.action_to_id}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> ActionVocab:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls()
        vocab.action_to_id = {str(k): int(v) for k, v in payload["action_to_id"].items()}
        vocab.id_to_action = {v: k for k, v in vocab.action_to_id.items()}
        return vocab

    @classmethod
    def build_from_records(cls, records: list[dict[str, Any]]) -> ActionVocab:
        vocab = cls()
        for record in records:
            action = record.get("action")
            if action:
                vocab.encode(action)
        return vocab

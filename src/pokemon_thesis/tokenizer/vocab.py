from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


PAD = "<pad>"
UNK = "<unk>"


class TokenVocab:
    """Maps canonical token strings to integer ids."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {PAD: 0, UNK: 1}
        self.id_to_token: dict[int, str] = {0: PAD, 1: UNK}

    def __len__(self) -> int:
        return len(self.token_to_id)

    def add(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        next_id = len(self.token_to_id)
        self.token_to_id[token] = next_id
        self.id_to_token[next_id] = token
        return next_id

    def encode(self, token: str) -> int:
        if token not in self.token_to_id:
            self.add(token)
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, UNK)

    def lookup(self, token: str) -> int:
        return self.token_to_id.get(token, 1)

    def encode_or_unk(self, token: str) -> int:
        return self.lookup(token)

    def tokens(self) -> Iterator[str]:
        return iter(self.token_to_id.keys())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"token_to_id": self.token_to_id}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TokenVocab:
        payload = json.loads(path.read_text(encoding="utf-8"))
        vocab = cls()
        vocab.token_to_id = {str(k): int(v) for k, v in payload["token_to_id"].items()}
        vocab.id_to_token = {v: k for k, v in vocab.token_to_id.items()}
        return vocab

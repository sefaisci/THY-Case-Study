"""Deterministic local sparse vectors used for ingestion and querying."""

from __future__ import annotations

import hashlib
import re

from qdrant_client import models

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class StableHashSparseEncoder:
    """Map lexical token frequencies into a stable large hash space."""

    version = "blake2b-frequency-v1"
    hash_space = 2_000_000_000

    def encode(self, text: str) -> models.SparseVector:
        """Encode text as deterministic sorted sparse indices and frequencies."""

        frequencies: dict[int, float] = {}
        for token in TOKEN_PATTERN.findall(text.lower()):
            index = self._index(token)
            frequencies[index] = frequencies.get(index, 0.0) + 1.0
        if not frequencies:
            frequencies[self._index("__empty__")] = 1.0
        ordered = sorted(frequencies.items())
        return models.SparseVector(
            indices=[index for index, _ in ordered],
            values=[value for _, value in ordered],
        )

    def _index(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % self.hash_space

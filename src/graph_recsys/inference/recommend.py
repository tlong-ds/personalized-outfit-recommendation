from __future__ import annotations

from typing import Sequence


def recommend_top_k(user_id: str, candidates: Sequence[str], k: int = 10) -> list[str]:
    _ = user_id
    return list(candidates)[:k]

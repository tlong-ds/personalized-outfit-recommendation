from __future__ import annotations

import hashlib
import random
from typing import Iterable

import numpy as np


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    val = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
    return val % modulo


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def seeded_choice(seq: list[str], k: int, seed_text: str) -> list[str]:
    if k >= len(seq):
        return seq
    rng = random.Random(stable_int(seed_text))
    return rng.sample(seq, k)


def hash_vector(key: str, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(stable_int(f"{seed}:{key}"))
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def mean_pool(vectors: Iterable[np.ndarray]) -> np.ndarray:
    vectors = list(vectors)
    if not vectors:
        raise ValueError("cannot pool empty vectors")
    out = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(out) + 1e-12
    return (out / norm).astype(np.float32)

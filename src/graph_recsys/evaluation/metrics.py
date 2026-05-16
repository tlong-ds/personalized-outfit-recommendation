from __future__ import annotations

import math
from typing import Iterable


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def auc_roc(y_true: list[int], y_score: list[float]) -> float:
    # Mann-Whitney U equivalent.
    pos = [(s, y) for s, y in zip(y_score, y_true) if y == 1]
    neg = [(s, y) for s, y in zip(y_score, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    combined = sorted([(s, y) for s, y in zip(y_score, y_true)], key=lambda x: x[0])
    ranks = {}
    for i, (s, y) in enumerate(combined, start=1):
        ranks.setdefault((s, y), []).append(i)
    rank_sum = 0.0
    for i, (s, y) in enumerate(combined, start=1):
        if y == 1:
            rank_sum += i
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def ndcg_at_k(hit_rank: int, k: int) -> float:
    if hit_rank <= 0 or hit_rank > k:
        return 0.0
    return 1.0 / math.log2(hit_rank + 1)

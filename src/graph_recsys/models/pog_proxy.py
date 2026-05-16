from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graph_recsys.evaluation.metrics import ndcg_at_k
from graph_recsys.utils.io import write_json


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in str(text).split()], dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def evaluate_pog_proxy(
    embedding_csv: Path,
    canonical_dir: Path,
    out_json: Path,
    k: int = 10,
    user_limit: int = 2000,
    candidate_pool_limit: int = 10000,
) -> dict[str, float]:
    emb_df = pd.read_csv(embedding_csv)
    emb = {row.item_id: _parse_vec(row.emb_image_text_cf) for row in emb_df.itertuples(index=False)}

    users = pd.read_csv(canonical_dir / "user_sequences.csv")
    outfits = pd.read_csv(canonical_dir / "outfits.csv")
    outfit_map = {
        r.outfit_id: [x for x in str(r.item_seq).split(";") if x] for r in outfits.itertuples(index=False)
    }
    item_pool = sorted(emb.keys())
    if len(item_pool) > candidate_pool_limit:
        step = max(1, len(item_pool) // candidate_pool_limit)
        item_pool = item_pool[::step][:candidate_pool_limit]

    hit = 0
    total = 0
    mrr = 0.0
    ndcg = 0.0

    for row in users.head(user_limit).itertuples(index=False):
        clicks = [x for x in str(row.click_seq).split(';') if x in emb]
        if len(clicks) < 3:
            continue
        user_vec = np.mean(np.stack([emb[x] for x in clicks[-20:]]), axis=0)
        scores = sorted(((it, _cos(user_vec, emb[it])) for it in item_pool), key=lambda x: x[1], reverse=True)[:k]
        topk = [it for it, _ in scores]

        true_items: list[str]
        target_outfit_id = getattr(row, "target_outfit_id", "")
        if target_outfit_id and target_outfit_id in outfit_map:
            true_items = [x for x in outfit_map[target_outfit_id] if x in emb]
        else:
            true_items = [clicks[-1]]

        total += 1
        ranks = [topk.index(x) + 1 for x in true_items if x in topk]
        if ranks:
            best_rank = min(ranks)
            hit += 1
            mrr += 1.0 / best_rank
            ndcg += ndcg_at_k(best_rank, k)

    res = {
        "hit_rate_at_k": hit / max(1, total),
        "mrr_at_k": mrr / max(1, total),
        "ndcg_at_k": ndcg / max(1, total),
        "evaluated_users": total,
        "candidate_pool_size": len(item_pool),
        "notes": "Offline personalization proxy (not Dida CTR).",
    }
    write_json(out_json, res)
    return res

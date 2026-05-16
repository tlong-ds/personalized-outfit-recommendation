from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graph_recsys.utils.io import ensure_dir, write_json
from graph_recsys.utils.random import hash_vector, mean_pool


def _text_vector(title: str, dim: int, seed: int) -> np.ndarray:
    toks = [tok.strip().lower() for tok in title.split() if tok.strip()]
    if not toks:
        return hash_vector("[empty-text]", dim=dim, seed=seed)
    return mean_pool(hash_vector(f"tok::{tok}", dim=dim, seed=seed) for tok in toks[:32])


def train_embeddings(canonical_dir: Path, artifacts_dir: Path, de: int, seed: int) -> dict[str, str]:
    ensure_dir(artifacts_dir)
    items = pd.read_csv(canonical_dir / "items.csv")

    image_dim = 1536
    text_dim = 300
    cf_dim = 160

    out_rows = []
    for row in items.itertuples(index=False):
        image128 = hash_vector(f"proj-img::{row.image_url}", dim=de, seed=seed + 3)
        text128 = _text_vector(str(row.title), dim=de, seed=seed + 13)
        cf128 = hash_vector(f"proj-cf::{row.item_id}", dim=de, seed=seed + 7)

        fused = mean_pool([image128, text128, cf128])
        image_text = mean_pool([image128, text128])

        out_rows.append(
            {
                "item_id": row.item_id,
                "category_id": row.category_id,
                "emb_image": " ".join(map(str, image128.tolist())),
                "emb_text": " ".join(map(str, text128.tolist())),
                "emb_cf": " ".join(map(str, cf128.tolist())),
                "emb_image_text": " ".join(map(str, image_text.tolist())),
                "emb_image_text_cf": " ".join(map(str, fused.tolist())),
                "raw_dims": f"img={image_dim},text={text_dim},cf={cf_dim}",
            }
        )

    out_csv = artifacts_dir / "item_embeddings.csv"
    pd.DataFrame(out_rows).to_csv(out_csv, index=False)
    meta = {
        "embedding_csv": str(out_csv),
        "de": de,
        "notes": [
            "Pragmatic-faithful proxy: proprietary image/text/cf encoders are replaced by deterministic hashed embeddings.",
            "Ablation columns are compatible with paper-style modality comparisons.",
        ],
    }
    write_json(artifacts_dir / "embedding_meta.json", meta)
    return {"embedding_csv": str(out_csv)}

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from graph_recsys.utils.io import ensure_dir, write_json
from graph_recsys.utils.random import set_global_seed


class FusionProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.proj(x), dim=-1)


def _hash_vec(key: str, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((key, seed))) % (2**32))
    return rng.standard_normal(dim).astype(np.float32)


def _title_vec(title: str, dim: int, seed: int) -> np.ndarray:
    toks = [t.strip().lower() for t in str(title).split() if t.strip()]
    if not toks:
        return _hash_vec("empty", dim, seed)
    mats = np.stack([_hash_vec(f"tok::{tok}", dim, seed) for tok in toks[:24]], axis=0)
    return mats.mean(axis=0)


def train_embeddings(
    canonical_dir: Path,
    artifacts_dir: Path,
    de: int,
    seed: int,
    epochs: int = 3,
    triplet_margin: float = 0.1,
) -> dict[str, str]:
    ensure_dir(artifacts_dir)
    set_global_seed(seed)
    torch.manual_seed(seed)

    items = pd.read_parquet(canonical_dir / "items.parquet")
    base_dim = 64

    image = np.stack([_hash_vec(f"img::{u}", base_dim, seed + 7) for u in items["image_url"].astype(str)], axis=0)
    text = np.stack([_title_vec(t, base_dim, seed + 11) for t in items["title"].astype(str)], axis=0)
    cf = np.stack([_hash_vec(f"cf::{i}", base_dim, seed + 17) for i in items["item_id"].astype(str)], axis=0)

    x = torch.tensor(np.concatenate([image, text, cf], axis=1), dtype=torch.float32)
    cats = items["category_id"].astype(str).tolist()
    model = FusionProjector(in_dim=x.shape[1], out_dim=de)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    triplet = nn.TripletMarginLoss(margin=triplet_margin, p=2)

    for _ in range(epochs):
        model.train()
        emb = model(x)
        losses: list[torch.Tensor] = []
        for i, cat in enumerate(cats):
            pos = next((j for j, c in enumerate(cats) if c == cat and j != i), None)
            neg = next((j for j, c in enumerate(cats) if c != cat), None)
            if pos is None or neg is None:
                continue
            losses.append(triplet(emb[i : i + 1], emb[pos : pos + 1], emb[neg : neg + 1]))
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        fused = model(x).cpu().numpy()
        image_p = nn.functional.normalize(torch.tensor(image)[:, :de], dim=-1).numpy()
        text_p = nn.functional.normalize(torch.tensor(text)[:, :de], dim=-1).numpy()
        cf_p = nn.functional.normalize(torch.tensor(cf)[:, :de], dim=-1).numpy()
        image_text = nn.functional.normalize((torch.tensor(image_p) + torch.tensor(text_p)) / 2, dim=-1).numpy()

    out = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "category_id": items["category_id"],
            "emb_image": [" ".join(map(str, v.tolist())) for v in image_p],
            "emb_text": [" ".join(map(str, v.tolist())) for v in text_p],
            "emb_cf": [" ".join(map(str, v.tolist())) for v in cf_p],
            "emb_image_text": [" ".join(map(str, v.tolist())) for v in image_text],
            "emb_image_text_cf": [" ".join(map(str, v.tolist())) for v in fused],
        }
    )
    out_csv = artifacts_dir / "item_embeddings.parquet"
    out.to_parquet(out_csv, index=False)
    ckpt = artifacts_dir / "embedding_model.pt"
    torch.save(model.state_dict(), ckpt)
    write_json(
        artifacts_dir / "embedding_meta.json",
        {"embedding_csv": str(out_csv), "checkpoint": str(ckpt), "de": de, "triplet_margin": triplet_margin},
    )
    return {"embedding_csv": str(out_csv), "checkpoint": str(ckpt)}

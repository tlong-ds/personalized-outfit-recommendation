from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from graph_recsys.evaluation.metrics import ndcg_at_k
from graph_recsys.utils.io import read_json, write_json


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in str(text).split()], dtype=np.float32)


def _seq(x: str) -> list[str]:
    return [s for s in str(x).split(";") if s]


class POGModel(nn.Module):
    def __init__(self, de: int, dm: int, layers: int, heads: int, n_items: int) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(de, dm)
        self.dec_proj = nn.Linear(de, dm)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dm, nhead=heads, batch_first=True), num_layers=layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dm, nhead=heads, batch_first=True), num_layers=layers
        )
        self.out = nn.Linear(dm, n_items)

    def forward(self, click_emb: torch.Tensor, out_emb: torch.Tensor) -> torch.Tensor:
        mem = self.encoder(self.enc_proj(click_emb))
        dec = self.decoder(self.dec_proj(out_emb), mem)
        return self.out(dec[:, -1, :])


def should_stop_decode(token: str, length: int, end_token: str = "[END]", max_len: int = 8) -> bool:
    return token == end_token or length >= max_len


def train_pog(
    embedding_csv: Path,
    canonical_dir: Path,
    processed_dir: Path,
    checkpoint_path: Path,
    emb_col: str,
    de: int,
    dm: int,
    layers: int,
    heads: int,
    epochs: int,
) -> Path:
    emb_df = pd.read_parquet(embedding_csv)
    item_ids = emb_df["item_id"].astype(str).tolist()
    idx = {x: i for i, x in enumerate(item_ids)}
    table = np.stack([_parse_vec(v) for v in emb_df[emb_col].astype(str)], axis=0)
    users = pd.read_parquet(canonical_dir / "user_sequences.parquet")
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    outfit_map = {r.outfit_id: _seq(r.item_seq) for r in outfits.itertuples(index=False)}
    train_users = set(read_json(processed_dir / "split_manifest.json")["pog"]["train_user_ids"])
    data = users[users["user_id"].isin(train_users)]

    model = POGModel(de=de, dm=dm, layers=layers, heads=heads, n_items=len(item_ids))
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        for row in data.itertuples(index=False):
            clicks = [x for x in _seq(row.click_seq) if x in idx]
            target = outfit_map.get(getattr(row, "target_outfit_id", ""), [])
            target = [x for x in target if x in idx]
            if len(clicks) < 2 or len(target) < 2:
                continue
            click_x = torch.tensor(np.stack([table[idx[c]] for c in clicks[-20:]], axis=0), dtype=torch.float32).unsqueeze(0)
            for step in range(1, len(target)):
                dec_in = target[:step]
                y = target[step]
                dec_x = torch.tensor(np.stack([table[idx[c]] for c in dec_in], axis=0), dtype=torch.float32).unsqueeze(0)
                logits = model(click_x, dec_x)
                loss = ce(logits, torch.tensor([idx[y]], dtype=torch.long))
                opt.zero_grad()
                loss.backward()
                opt.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def evaluate_pog_proxy(
    embedding_csv: Path,
    canonical_dir: Path,
    out_json: Path,
    k: int = 10,
    user_limit: int = 2000,
    candidate_pool_limit: int = 10000,
) -> dict[str, float]:
    emb_df = pd.read_parquet(embedding_csv)
    emb = {row.item_id: _parse_vec(row.emb_image_text_cf) for row in emb_df.itertuples(index=False)}

    users = pd.read_parquet(canonical_dir / "user_sequences.parquet")
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    outfit_map = {r.outfit_id: _seq(r.item_seq) for r in outfits.itertuples(index=False)}
    item_pool = sorted(emb.keys())[:candidate_pool_limit]

    hit = 0
    total = 0
    mrr = 0.0
    ndcg = 0.0
    for row in users.head(user_limit).itertuples(index=False):
        clicks = [x for x in _seq(row.click_seq) if x in emb]
        if len(clicks) < 2:
            continue
        user_vec = np.mean(np.stack([emb[x] for x in clicks[-20:]], axis=0), axis=0)
        ranked = sorted(((it, float(np.dot(user_vec, emb[it]))) for it in item_pool), key=lambda x: x[1], reverse=True)[:k]
        topk = [it for it, _ in ranked]
        true_items = [x for x in outfit_map.get(getattr(row, "target_outfit_id", ""), []) if x in emb]
        if not true_items:
            true_items = [clicks[-1]]
        total += 1
        ranks = [topk.index(x) + 1 for x in true_items if x in topk]
        if ranks:
            best = min(ranks)
            hit += 1
            mrr += 1.0 / best
            ndcg += ndcg_at_k(best, k)

    res = {
        "hit_rate_at_k": hit / max(1, total),
        "mrr_at_k": mrr / max(1, total),
        "ndcg_at_k": ndcg / max(1, total),
        "evaluated_users": total,
    }
    write_json(out_json, res)
    return res

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from graph_recsys.evaluation.metrics import accuracy, auc_roc
from graph_recsys.utils.io import read_json, write_json


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in str(text).split()], dtype=np.float32)


def _seq(x: str) -> list[str]:
    return [s for s in str(x).split(";") if s]


class FOMModel(nn.Module):
    def __init__(self, de: int, dm: int, layers: int, heads: int, n_items: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(de, dm)
        enc_layer = nn.TransformerEncoderLayer(d_model=dm, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Linear(dm, n_items)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.out(pooled)


def sample_masked_sequence(seq: list[str]) -> tuple[list[str], str, int]:
    if len(seq) < 2:
        raise ValueError("sequence must contain at least 2 items")
    pos = len(seq) // 2
    target = seq[pos]
    context = [x for i, x in enumerate(seq) if i != pos]
    return context, target, pos


def _build_item_table(embedding_csv: Path, emb_col: str) -> tuple[list[str], dict[str, int], np.ndarray]:
    emb_df = pd.read_parquet(embedding_csv)
    item_ids = emb_df["item_id"].astype(str).tolist()
    vec = np.stack([_parse_vec(v) for v in emb_df[emb_col].astype(str)], axis=0)
    return item_ids, {x: i for i, x in enumerate(item_ids)}, vec


def train_fom(
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
    item_ids, item_to_idx, table = _build_item_table(embedding_csv, emb_col)
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    train_ids = set(read_json(processed_dir / "split_manifest.json")["fom"]["train_outfit_ids"])
    train = outfits[outfits["outfit_id"].isin(train_ids)]

    model = FOMModel(de=de, dm=dm, layers=layers, heads=heads, n_items=len(item_ids))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for row in train.itertuples(index=False):
            seq = [x for x in _seq(row.item_seq) if x in item_to_idx]
            if len(seq) < 2:
                continue
            context, target, _ = sample_masked_sequence(seq)
            x = torch.tensor(np.stack([table[item_to_idx[s]] for s in context], axis=0), dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            y = torch.tensor([item_to_idx[target]], dtype=torch.long)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def evaluate_fom(
    embedding_csv: Path,
    processed_dir: Path,
    out_json: Path,
    emb_col: str = "emb_image_text_cf",
) -> dict[str, float]:
    emb_df = pd.read_parquet(embedding_csv)
    emb = {row.item_id: _parse_vec(getattr(row, emb_col)) for row in emb_df.itertuples(index=False)}

    fitb = pd.read_parquet(processed_dir / "fitb_eval.parquet")
    hits = []
    for row in fitb.itertuples(index=False):
        ctx = [emb[x] for x in _seq(row.context_items) if x in emb]
        cands = [x for x in _seq(row.choice_items) if x in emb]
        if not ctx or not cands:
            continue
        mean = np.mean(np.stack(ctx), axis=0)
        scores = [(c, float(np.dot(mean, emb[c]))) for c in cands]
        pred = max(scores, key=lambda x: x[1])[0]
        hits.append(int(pred == row.true_item))
    fitb_acc = accuracy([1] * len(hits), hits)

    cp = pd.read_parquet(processed_dir / "cp_eval.parquet")
    y_true: list[int] = []
    y_score: list[float] = []
    for row in cp.itertuples(index=False):
        seq = [x for x in _seq(row.item_seq) if x in emb]
        if len(seq) < 2:
            continue
        vecs = [emb[x] for x in seq]
        score = float(np.mean([np.dot(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]))
        y_true.append(int(row.label))
        y_score.append(score)
    cp_auc = auc_roc(y_true, y_score)

    res = {"fitb_accuracy": float(fitb_acc), "cp_auc": float(cp_auc)}
    write_json(out_json, res)
    return res

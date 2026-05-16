from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from graph_recsys.evaluation.metrics import accuracy, auc_roc
from graph_recsys.utils.io import write_json


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in str(text).split()], dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _outfit_score(item_ids: list[str], emb: dict[str, np.ndarray]) -> float:
    vecs = [emb[x] for x in item_ids if x in emb]
    if len(vecs) < 2:
        return 0.0
    s = 0.0
    n = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s += _cos(vecs[i], vecs[j])
            n += 1
    return s / max(1, n)


def evaluate_fom(embedding_csv: Path, processed_dir: Path, out_json: Path, emb_col: str = "emb_image_text_cf") -> dict[str, float]:
    emb_df = pd.read_csv(embedding_csv)
    emb = {row.item_id: _parse_vec(getattr(row, emb_col)) for row in emb_df.itertuples(index=False)}

    fitb = pd.read_csv(processed_dir / "fitb_eval.csv")
    fitb_true = []
    fitb_pred = []
    for row in fitb.itertuples(index=False):
        context = [x for x in str(row.context_items).split(";") if x]
        cands = [x for x in str(row.choice_items).split(";") if x]
        cvecs = [emb[x] for x in context if x in emb]
        if not cvecs:
            continue
        ctx = np.mean(np.stack(cvecs), axis=0)
        best = None
        best_score = -1e9
        for c in cands:
            if c not in emb:
                continue
            score = _cos(ctx, emb[c])
            if score > best_score:
                best_score = score
                best = c
        fitb_true.append(row.true_item)
        fitb_pred.append(best)

    fitb_acc = accuracy([1] * len(fitb_true), [int(t == p) for t, p in zip(fitb_true, fitb_pred)])

    cp = pd.read_csv(processed_dir / "cp_eval.csv")
    y_true = []
    y_score = []
    for row in cp.itertuples(index=False):
        seq = [x for x in str(row.item_seq).split(";") if x]
        y_true.append(int(row.label))
        y_score.append(_outfit_score(seq, emb))
    cp_auc = auc_roc(y_true, y_score)

    res = {"fitb_accuracy": float(fitb_acc), "cp_auc": float(cp_auc)}
    write_json(out_json, res)
    return res

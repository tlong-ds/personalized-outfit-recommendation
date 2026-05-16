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


def _prepare_embeddings(path: Path, emb_col: str) -> dict[str, np.ndarray]:
    df = pd.read_csv(path)
    return {row.item_id: _parse_vec(getattr(row, emb_col)) for row in df.itertuples(index=False)}


def _ordered(seq: list[str]) -> list[str]:
    return sorted(seq)


def _score_setnn(context: list[str], cand: str, emb: dict[str, np.ndarray]) -> float:
    ctx = [emb[x] for x in context if x in emb]
    if not ctx or cand not in emb:
        return -1e9
    c = np.mean(np.stack(ctx), axis=0)
    return _cos(c, emb[cand])


def _score_flstm(context: list[str], cand: str, emb: dict[str, np.ndarray], ordered: bool) -> float:
    seq = _ordered(context) if ordered else context
    if not seq or cand not in emb:
        return -1e9
    last = emb[seq[-1]] if seq[-1] in emb else None
    if last is None:
        return -1e9
    return _cos(last, emb[cand])


def _score_bilstm(context: list[str], cand: str, emb: dict[str, np.ndarray], ordered: bool) -> float:
    seq = _ordered(context) if ordered else context
    if not seq or cand not in emb:
        return -1e9
    ref = []
    if seq[0] in emb:
        ref.append(emb[seq[0]])
    if seq[-1] in emb:
        ref.append(emb[seq[-1]])
    if not ref:
        return -1e9
    mean_ref = np.mean(np.stack(ref), axis=0)
    return _cos(mean_ref, emb[cand])


def run_baselines(embedding_csv: Path, processed_dir: Path, out_json: Path, emb_col: str = "emb_image_text_cf") -> dict[str, dict[str, float]]:
    emb = _prepare_embeddings(embedding_csv, emb_col)
    fitb = pd.read_csv(processed_dir / "fitb_eval.csv")
    cp = pd.read_csv(processed_dir / "cp_eval.csv")

    models = {
        "f_lstm_unordered": lambda ctx, cand: _score_flstm(ctx, cand, emb, ordered=False),
        "f_lstm_ordered": lambda ctx, cand: _score_flstm(ctx, cand, emb, ordered=True),
        "bi_lstm_unordered": lambda ctx, cand: _score_bilstm(ctx, cand, emb, ordered=False),
        "bi_lstm_ordered": lambda ctx, cand: _score_bilstm(ctx, cand, emb, ordered=True),
        "setnn_unordered": lambda ctx, cand: _score_setnn(ctx, cand, emb),
        "setnn_ordered": lambda ctx, cand: _score_setnn(_ordered(ctx), cand, emb),
    }

    out: dict[str, dict[str, float]] = {}
    for name, scorer in models.items():
        fitb_hits = []
        for row in fitb.itertuples(index=False):
            context = [x for x in str(row.context_items).split(";") if x]
            cands = [x for x in str(row.choice_items).split(";") if x]
            scored = [(c, scorer(context, c)) for c in cands]
            best = max(scored, key=lambda x: x[1])[0] if scored else ""
            fitb_hits.append(int(best == row.true_item))

        y_true = []
        y_score = []
        for row in cp.itertuples(index=False):
            seq = [x for x in str(row.item_seq).split(";") if x]
            if len(seq) < 2:
                continue
            ref = seq[:-1]
            target = seq[-1]
            y_true.append(int(row.label))
            y_score.append(scorer(ref, target))

        out[name] = {
            "fitb_accuracy": accuracy([1] * len(fitb_hits), fitb_hits),
            "cp_auc": auc_roc(y_true, y_score),
        }

    write_json(out_json, out)
    return out

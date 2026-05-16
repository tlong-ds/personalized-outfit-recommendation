from pathlib import Path

from graph_recsys.data.canonicalize import canonicalize_raw
from graph_recsys.data.splits import build_split_manifest
from graph_recsys.embeddings.train import train_embeddings
from graph_recsys.evaluation.build_sets import build_cp_set, build_fitb_set
from graph_recsys.models.baselines import run_baselines
from graph_recsys.models.fom import evaluate_fom
from graph_recsys.models.pog_proxy import evaluate_pog_proxy


def _write_raw(root: Path) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    (raw / "item_data.txt").write_text(
        "i1,c1,url1,red shirt\n"
        "i2,c2,url2,blue jeans\n"
        "i3,c3,url3,white shoes\n"
        "i4,c4,url4,black bag\n"
        "i5,c2,url5,green pants\n",
        encoding="utf-8",
    )
    (raw / "outfit_data.txt").write_text(
        "o1,i1;i2;i3\n"
        "o2,i1;i5;i3\n"
        "o3,i4;i2;i3\n"
        "o4,i4;i5;i3\n",
        encoding="utf-8",
    )
    (raw / "user_data.txt").write_text(
        "u1,i1;i2;i3,o1\n"
        "u2,i4;i5;i3,o4\n"
        "u3,i1;i5;i3,o2\n",
        encoding="utf-8",
    )


def test_end_to_end_smoke(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    canonical = tmp_path / "canonical"
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    processed.mkdir(parents=True, exist_ok=True)

    stats = canonicalize_raw(tmp_path / "raw", canonical, sample_ratio=1.0)
    assert stats.items == 5

    build_split_manifest(canonical, processed, test_ratio=0.25, seed=42)
    build_fitb_set(canonical, processed)
    build_cp_set(canonical, processed)

    train_embeddings(canonical, artifacts, de=32, seed=42)
    fom = evaluate_fom(artifacts / "item_embeddings.csv", processed, artifacts / "fom.json")
    baselines = run_baselines(artifacts / "item_embeddings.csv", processed, artifacts / "base.json")
    pog = evaluate_pog_proxy(artifacts / "item_embeddings.csv", canonical, artifacts / "pog.json", k=3)

    assert 0.0 <= fom["fitb_accuracy"] <= 1.0
    assert 0.0 <= fom["cp_auc"] <= 1.0
    assert "f_lstm_unordered" in baselines
    assert 0.0 <= pog["hit_rate_at_k"] <= 1.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from graph_recsys.utils.io import write_json
from graph_recsys.utils.random import set_global_seed


def build_split_manifest(canonical_dir: Path, processed_dir: Path, test_ratio: float, seed: int) -> Path:
    outfits = pd.read_csv(canonical_dir / "outfits.csv")
    set_global_seed(seed)

    shuffled = outfits.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(len(shuffled) * test_ratio))

    test_ids = shuffled.iloc[:n_test]["outfit_id"].tolist()
    train_ids = shuffled.iloc[n_test:]["outfit_id"].tolist()

    manifest = {
        "seed": seed,
        "test_ratio": test_ratio,
        "train_outfit_count": len(train_ids),
        "test_outfit_count": len(test_ids),
        "train_outfit_ids": train_ids,
        "test_outfit_ids": test_ids,
    }
    out_path = processed_dir / "split_manifest.json"
    write_json(out_path, manifest)
    return out_path

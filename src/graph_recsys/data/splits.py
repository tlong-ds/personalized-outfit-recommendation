from __future__ import annotations

from pathlib import Path

import pandas as pd

from graph_recsys.utils.io import write_json
from graph_recsys.utils.random import set_global_seed


def build_split_manifest(canonical_dir: Path, processed_dir: Path, test_ratio: float, seed: int) -> Path:
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    users = pd.read_parquet(canonical_dir / "user_sequences.parquet")
    set_global_seed(seed)

    shuffled = outfits.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = max(1, int(len(shuffled) * test_ratio))
    n_val = max(1, int(len(shuffled) * test_ratio))

    test_ids = shuffled.iloc[:n_test]["outfit_id"].tolist()
    val_ids = shuffled.iloc[n_test : n_test + n_val]["outfit_id"].tolist()
    train_ids = shuffled.iloc[n_test + n_val :]["outfit_id"].tolist()

    shuffled_users = users.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    nu_test = max(1, int(len(shuffled_users) * test_ratio))
    nu_val = max(1, int(len(shuffled_users) * test_ratio))
    pog_test = shuffled_users.iloc[:nu_test]["user_id"].tolist()
    pog_val = shuffled_users.iloc[nu_test : nu_test + nu_val]["user_id"].tolist()
    pog_train = shuffled_users.iloc[nu_test + nu_val :]["user_id"].tolist()

    manifest = {
        "seed": seed,
        "test_ratio": test_ratio,
        "fom": {
            "train_outfit_ids": train_ids,
            "val_outfit_ids": val_ids,
            "test_outfit_ids": test_ids,
        },
        "pog": {
            "train_user_ids": pog_train,
            "val_user_ids": pog_val,
            "test_user_ids": pog_test,
        },
    }
    out_path = processed_dir / "split_manifest.json"
    write_json(out_path, manifest)
    return out_path

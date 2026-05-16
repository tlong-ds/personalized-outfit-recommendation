from __future__ import annotations

from pathlib import Path

import pandas as pd


def prepare_interactions(raw_dir: Path, user_file: str, item_file: str, output_path: Path) -> pd.DataFrame:
    users = pd.read_csv(raw_dir / user_file, sep=None, engine="python")
    items = pd.read_csv(raw_dir / item_file, sep=None, engine="python")

    # Baseline synthetic interaction frame to unblock graph pipeline setup.
    users = users.reset_index(drop=True)
    items = items.reset_index(drop=True)
    n = min(len(users), len(items))

    interactions = pd.DataFrame(
        {
            "user_id": users.iloc[:n, 0].astype(str).values,
            "item_id": items.iloc[:n, 0].astype(str).values,
            "weight": 1.0,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(output_path, index=False)
    return interactions

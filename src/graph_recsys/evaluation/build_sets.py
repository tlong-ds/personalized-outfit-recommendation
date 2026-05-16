from __future__ import annotations

from pathlib import Path

import pandas as pd

from graph_recsys.utils.io import ensure_dir, read_json
from graph_recsys.utils.random import seeded_choice


def _parse_seq(seq: str) -> list[str]:
    return [x for x in str(seq).split(";") if x]


def build_fitb_set(canonical_dir: Path, processed_dir: Path, negative_k: int = 3) -> Path:
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    manifest = read_json(processed_dir / "split_manifest.json")
    test_ids = set(manifest["fom"]["test_outfit_ids"])

    pool = set()
    for seq in outfits["item_seq"].astype(str):
        pool.update(_parse_seq(seq))
    pool_list = sorted(pool)

    rows = []
    test_outfits = outfits[outfits["outfit_id"].isin(test_ids)]
    for row in test_outfits.itertuples(index=False):
        seq = _parse_seq(row.item_seq)
        for pos, true_item in enumerate(seq):
            context = [x for i, x in enumerate(seq) if i != pos]
            candidates = [x for x in pool_list if x != true_item and x not in seq]
            negs = seeded_choice(candidates, negative_k, seed_text=f"fitb:{row.outfit_id}:{pos}")
            choice_set = [true_item] + negs
            rows.append(
                {
                    "outfit_id": row.outfit_id,
                    "mask_position": pos,
                    "true_item": true_item,
                    "context_items": ";".join(context),
                    "choice_items": ";".join(choice_set),
                }
            )

    ensure_dir(processed_dir)
    out_path = processed_dir / "fitb_eval.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path


def build_cp_set(canonical_dir: Path, processed_dir: Path) -> Path:
    outfits = pd.read_parquet(canonical_dir / "outfits.parquet")
    manifest = read_json(processed_dir / "split_manifest.json")
    test_ids = set(manifest["fom"]["test_outfit_ids"])

    compatible = outfits[outfits["outfit_id"].isin(test_ids)].copy()

    pool = set()
    for seq in outfits["item_seq"].astype(str):
        pool.update(_parse_seq(seq))
    pool_list = sorted(pool)

    incompatible_rows = []
    for row in compatible.itertuples(index=False):
        seq = _parse_seq(row.item_seq)
        if len(seq) < 2:
            continue
        replace_pos = hash(f"cp:{row.outfit_id}") % len(seq)
        repl_candidates = [x for x in pool_list if x not in seq]
        if not repl_candidates:
            continue
        repl = repl_candidates[hash(f"cp-repl:{row.outfit_id}") % len(repl_candidates)]
        seq2 = seq.copy()
        seq2[replace_pos] = repl
        incompatible_rows.append({"outfit_id": f"neg::{row.outfit_id}", "item_seq": ";".join(seq2), "label": 0})

    comp_rows = compatible[["outfit_id", "item_seq"]].copy()
    comp_rows["label"] = 1

    cp = pd.concat([comp_rows, pd.DataFrame(incompatible_rows)], ignore_index=True)
    out_path = processed_dir / "cp_eval.parquet"
    cp.to_parquet(out_path, index=False)
    return out_path

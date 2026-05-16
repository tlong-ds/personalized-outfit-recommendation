from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from graph_recsys.utils.io import ensure_dir, write_json
from graph_recsys.utils.random import stable_int


@dataclass
class CanonicalStats:
    users: int
    items: int
    outfits: int
    outfit_edges: int
    user_click_edges: int


def _keep_by_ratio(key: str, sample_ratio: float) -> bool:
    if sample_ratio >= 1.0:
        return True
    v = stable_int(key, modulo=10_000)
    return v < int(sample_ratio * 10_000)


def canonicalize_raw(
    raw_dir: Path,
    out_dir: Path,
    sample_ratio: float,
    max_user_clicks: int = 50,
    max_items: int = 250_000,
) -> CanonicalStats:
    ensure_dir(out_dir)

    items_out = out_dir / "items.csv"
    outfits_out = out_dir / "outfits.csv"
    outfit_edges_out = out_dir / "outfit_item_edges.csv"
    user_seq_out = out_dir / "user_sequences.csv"
    user_click_edges_out = out_dir / "user_click_edges.csv"

    sampled_outfits_raw: list[tuple[str, list[str]]] = []
    required_item_ids: set[str] = set()
    required_item_ids_from_outfits: set[str] = set()
    with (raw_dir / "outfit_data.txt").open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "," not in line:
                continue
            outfit_id, item_seq = line.split(",", 1)
            if not _keep_by_ratio(outfit_id, sample_ratio):
                continue
            seq = [x for x in item_seq.split(";") if x]
            if len(seq) < 2:
                continue
            sampled_outfits_raw.append((outfit_id, seq))
            required_item_ids.update(seq)
            required_item_ids_from_outfits.update(seq)

    sampled_users_raw: list[tuple[str, list[str], str]] = []
    with (raw_dir / "user_data.txt").open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "," not in line:
                continue
            parts = line.split(",", 2)
            if len(parts) == 3:
                user_id, click_seq, target_outfit_id = parts
            elif len(parts) == 2:
                user_id, click_seq = parts
                target_outfit_id = ""
            else:
                continue
            if not _keep_by_ratio(user_id, sample_ratio):
                continue
            clicks = [x for x in click_seq.split(";") if x][:max_user_clicks]
            if len(clicks) < 2:
                continue
            sampled_users_raw.append((user_id, clicks, target_outfit_id))
            required_item_ids.update(clicks)

    if len(required_item_ids) > max_items:
        extra_budget = max(0, max_items - len(required_item_ids_from_outfits))
        rest = sorted(required_item_ids - required_item_ids_from_outfits)
        selected_rest = {
            item_id for item_id in rest if _keep_by_ratio(f"item-cap:{item_id}", extra_budget / max(1, len(rest)))
        }
        if len(selected_rest) > extra_budget:
            selected_rest = set(sorted(selected_rest)[:extra_budget])
        required_item_ids = required_item_ids_from_outfits | selected_rest

    item_rows = []
    with (raw_dir / "item_data.txt").open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",", 3)
            if len(parts) < 4:
                continue
            item_id, category_id, image_url, title = parts
            if item_id not in required_item_ids:
                continue
            item_rows.append((item_id, category_id, image_url, title))

    items = pd.DataFrame(item_rows, columns=["item_id", "category_id", "image_url", "title"])
    items.to_csv(items_out, index=False)
    item_set = set(items["item_id"].tolist())

    outfit_rows = []
    outfit_edges = []
    for outfit_id, seq_raw in sampled_outfits_raw:
        seq = [x for x in seq_raw if x in item_set]
        if len(seq) < 2:
            continue
        outfit_rows.append((outfit_id, len(seq), ";".join(seq)))
        for idx, item_id in enumerate(seq):
            outfit_edges.append((outfit_id, item_id, idx))

    outfits = pd.DataFrame(outfit_rows, columns=["outfit_id", "outfit_len", "item_seq"])
    outfit_edges_df = pd.DataFrame(outfit_edges, columns=["outfit_id", "item_id", "position"])
    outfits.to_csv(outfits_out, index=False)
    outfit_edges_df.to_csv(outfit_edges_out, index=False)

    user_seq_rows = []
    click_edges = []
    for user_id, clicks_raw, target_outfit_id in sampled_users_raw:
        clicks = [x for x in clicks_raw if x in item_set]
        if len(clicks) < 2:
            continue
        user_seq_rows.append((user_id, len(clicks), ";".join(clicks), target_outfit_id))
        for idx, item_id in enumerate(clicks):
            click_edges.append((user_id, item_id, idx))

    user_sequences = pd.DataFrame(
        user_seq_rows, columns=["user_id", "seq_len", "click_seq", "target_outfit_id"]
    )
    user_click_edges = pd.DataFrame(click_edges, columns=["user_id", "item_id", "position"])
    user_sequences.to_csv(user_seq_out, index=False)
    user_click_edges.to_csv(user_click_edges_out, index=False)

    stats = CanonicalStats(
        users=len(user_sequences),
        items=len(items),
        outfits=len(outfits),
        outfit_edges=len(outfit_edges_df),
        user_click_edges=len(user_click_edges),
    )
    write_json(
        out_dir / "canonical_stats.json",
        {
            "users": stats.users,
            "items": stats.items,
            "outfits": stats.outfits,
            "outfit_edges": stats.outfit_edges,
            "user_click_edges": stats.user_click_edges,
            "sample_ratio": sample_ratio,
            "notes": "Subset is deterministic by md5 hash thresholds.",
        },
    )
    write_json(out_dir / "outfit_map_meta.json", {"outfit_count": len(outfits)})
    return stats

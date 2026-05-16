from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.data.canonicalize import canonicalize_raw
from graph_recsys.data.splits import build_split_manifest
from graph_recsys.utils.config import load_config
from graph_recsys.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize raw iFashion data and build split manifest.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    ensure_dir(canonical_dir)
    ensure_dir(processed_dir)

    stats = canonicalize_raw(
        raw_dir=raw_dir,
        out_dir=canonical_dir,
        sample_ratio=float(cfg["data"]["sample_ratio"]),
        max_user_clicks=int(cfg["data"]["max_user_clicks"]),
        max_items=int(cfg["data"]["max_items"]),
    )
    manifest = build_split_manifest(
        canonical_dir=canonical_dir,
        processed_dir=processed_dir,
        test_ratio=float(cfg["data"]["test_ratio"]),
        seed=int(cfg["training"]["seed"]),
    )

    print(
        f"Canonicalized users={stats.users} items={stats.items} outfits={stats.outfits} "
        f"outfit_edges={stats.outfit_edges} user_click_edges={stats.user_click_edges}"
    )
    print(f"Split manifest: {manifest}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.evaluation.build_sets import build_cp_set, build_fitb_set
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FITB and CP evaluation sets.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    canonical_dir = Path(cfg["paths"]["canonical_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    fitb = build_fitb_set(canonical_dir=canonical_dir, processed_dir=processed_dir, negative_k=3)
    cp = build_cp_set(canonical_dir=canonical_dir, processed_dir=processed_dir)

    print(f"FITB set: {fitb}")
    print(f"CP set: {cp}")


if __name__ == "__main__":
    main()

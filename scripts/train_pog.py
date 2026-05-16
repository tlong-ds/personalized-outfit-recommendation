from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.models.pog_proxy import evaluate_pog_proxy
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline POG personalization proxy evaluation.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_json = Path(cfg["paths"]["artifacts_dir"]) / "pog_proxy_metrics.json"
    res = evaluate_pog_proxy(
        embedding_csv=Path(cfg["paths"]["artifacts_dir"]) / "item_embeddings.csv",
        canonical_dir=Path(cfg["paths"]["canonical_dir"]),
        out_json=out_json,
        k=int(cfg["evaluation"]["k"]),
        user_limit=int(cfg["evaluation"]["user_limit"]),
        candidate_pool_limit=int(cfg["evaluation"]["candidate_pool_limit"]),
    )
    print(res)


if __name__ == "__main__":
    main()

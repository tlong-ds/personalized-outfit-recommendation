from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.models.baselines import run_baselines
from graph_recsys.models.fom import evaluate_fom
from graph_recsys.models.pog_proxy import evaluate_pog_proxy
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation CLI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["fitb", "cp", "all", "pog"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts = Path(cfg["paths"]["artifacts_dir"])
    processed = Path(cfg["paths"]["processed_dir"])
    canonical = Path(cfg["paths"]["canonical_dir"])
    emb = artifacts / "item_embeddings.parquet"

    if args.task in {"fitb", "cp", "all"}:
        print(
            evaluate_fom(
                embedding_csv=emb,
                processed_dir=processed,
                out_json=artifacts / "fom_metrics.json",
                emb_col=cfg["evaluation"]["embedding_column"],
            )
        )
        print(
            run_baselines(
                embedding_csv=emb,
                processed_dir=processed,
                out_json=artifacts / "baseline_metrics.json",
                emb_col=cfg["evaluation"]["embedding_column"],
            )
        )

    if args.task in {"pog", "all"}:
        print(
            evaluate_pog_proxy(
                embedding_csv=emb,
                canonical_dir=canonical,
                out_json=artifacts / "pog_metrics.json",
                k=int(cfg["evaluation"]["k"]),
                user_limit=int(cfg["evaluation"]["user_limit"]),
                candidate_pool_limit=int(cfg["evaluation"]["candidate_pool_limit"]),
            )
        )


if __name__ == "__main__":
    main()

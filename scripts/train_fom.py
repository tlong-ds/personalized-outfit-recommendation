from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.models.fom import evaluate_fom
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FOM proxy on FITB/CP.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_json = Path(cfg["paths"]["artifacts_dir"]) / "fom_metrics.json"
    res = evaluate_fom(
        embedding_csv=Path(cfg["paths"]["artifacts_dir"]) / "item_embeddings.csv",
        processed_dir=Path(cfg["paths"]["processed_dir"]),
        out_json=out_json,
        emb_col=cfg["evaluation"]["embedding_column"],
    )
    print(res)


if __name__ == "__main__":
    main()

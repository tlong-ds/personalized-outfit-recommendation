from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.embeddings.train import train_embeddings
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pragmatic-faithful multimodal embedding proxies.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = train_embeddings(
        canonical_dir=Path(cfg["paths"]["canonical_dir"]),
        artifacts_dir=Path(cfg["paths"]["artifacts_dir"]),
        de=int(cfg["training"]["de"]),
        seed=int(cfg["training"]["seed"]),
    )
    print(f"Embedding artifact: {out['embedding_csv']}")


if __name__ == "__main__":
    main()

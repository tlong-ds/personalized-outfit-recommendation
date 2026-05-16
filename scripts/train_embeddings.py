from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pragmatic-faithful multimodal embedding proxies.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = str(cfg["training"].get("backend", "torch")).lower()
    kwargs = dict(
        canonical_dir=Path(cfg["paths"]["canonical_dir"]),
        artifacts_dir=Path(cfg["paths"]["artifacts_dir"]),
        de=int(cfg["training"]["de"]),
        seed=int(cfg["training"]["seed"]),
        epochs=int(cfg["training"]["epochs"]),
        triplet_margin=float(cfg["training"]["triplet_margin"]),
    )
    if backend == "tensorflow":
        from graph_recsys.training.tf_stack import train_embeddings_tf

        out = train_embeddings_tf(**kwargs)
    else:
        from graph_recsys.embeddings.train import train_embeddings

        out = train_embeddings(**kwargs)
    print(f"Embedding artifact: {out['embedding_csv']}")


if __name__ == "__main__":
    main()

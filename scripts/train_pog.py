from __future__ import annotations

from _bootstrap import setup_src_path
setup_src_path()

import argparse
from pathlib import Path

from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train POG encoder-decoder transformer.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    backend = str(cfg["training"].get("backend", "torch")).lower()
    kwargs = dict(
        embedding_csv=Path(cfg["paths"]["artifacts_dir"]) / "item_embeddings.parquet",
        canonical_dir=Path(cfg["paths"]["canonical_dir"]),
        processed_dir=Path(cfg["paths"]["processed_dir"]),
        checkpoint_path=Path(cfg["paths"]["artifacts_dir"]) / ("pog_model_tf.weights.h5" if backend == "tensorflow" else "pog_model.pt"),
        emb_col=cfg["evaluation"]["embedding_column"],
        de=int(cfg["training"]["de"]),
        dm=int(cfg["training"]["dm"]),
        layers=int(cfg["training"]["layers"]),
        heads=int(cfg["training"]["heads"]),
        epochs=int(cfg["training"]["epochs"]),
    )
    if backend == "tensorflow":
        from graph_recsys.training.tf_stack import train_pog_tf

        ckpt = train_pog_tf(**kwargs)
    else:
        from graph_recsys.models.pog_proxy import train_pog

        ckpt = train_pog(**kwargs)
    print(f"Saved POG checkpoint: {ckpt}")


if __name__ == "__main__":
    main()

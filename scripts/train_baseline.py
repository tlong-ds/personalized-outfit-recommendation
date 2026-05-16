from __future__ import annotations

import argparse
from pathlib import Path

from graph_recsys.training.train import train_baseline
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifact_path = train_baseline(
        artifacts_dir=Path(cfg["paths"]["artifacts_dir"]),
        epochs=int(cfg["training"]["epochs"]),
    )
    print(f"Saved model artifact: {artifact_path}")


if __name__ == "__main__":
    main()

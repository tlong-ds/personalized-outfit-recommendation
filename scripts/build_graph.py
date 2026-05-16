from __future__ import annotations

import argparse
from pathlib import Path

from graph_recsys.graph.build import build_bipartite_graph
from graph_recsys.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    interactions = Path(cfg["paths"]["interim_dir"]) / cfg["files"]["interactions"]
    out_edges = Path(cfg["paths"]["processed_dir"]) / cfg["files"]["graph_edges"]

    graph = build_bipartite_graph(interactions_path=interactions, edges_output_path=out_edges)
    print(f"Graph nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")


if __name__ == "__main__":
    main()

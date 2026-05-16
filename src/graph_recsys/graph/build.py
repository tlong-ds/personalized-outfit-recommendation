from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd


def build_bipartite_graph(interactions_path: Path, edges_output_path: Path) -> nx.Graph:
    df = pd.read_csv(interactions_path)

    graph = nx.Graph()
    for row in df.itertuples(index=False):
        user_node = f"u::{row.user_id}"
        item_node = f"i::{row.item_id}"
        graph.add_edge(user_node, item_node, weight=float(row.weight))

    edges_output_path.parent.mkdir(parents=True, exist_ok=True)
    edges = pd.DataFrame(graph.edges(data=True), columns=["src", "dst", "attr"])
    edges["weight"] = edges["attr"].apply(lambda x: x.get("weight", 1.0))
    edges[["src", "dst", "weight"]].to_csv(edges_output_path, index=False)
    return graph

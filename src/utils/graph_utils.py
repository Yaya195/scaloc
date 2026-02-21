import networkx as nx


def add_weighted_edge(graph: nx.Graph, u: int, v: int, weight: float) -> None:
    graph.add_edge(u, v, weight=weight)

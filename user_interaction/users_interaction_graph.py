from typing import Dict, Union, Callable, List

from user_interaction.user_interaction_parser import UsersInteraction

import networkx as nx

from matplotlib import pyplot as plt

DEFAULT_WEIGHT = 1.0


def get_one(*args, **kwargs): return DEFAULT_WEIGHT


def build_users_interaction_graph(users_interactions: Dict[str, Dict[str, UsersInteraction]],
                                  weight_func: Callable[[dict], float] = None
                                  ) -> nx.Graph:

    if weight_func is None:
        weight_func = get_one

    def dict_plus_weight(edge_data: dict, *args) -> dict:
        edge_data["weight"] = weight_func(edge_data, *args)
        return edge_data

    graph_data = []
    for source_user, target_users in users_interactions.items():
        source_user_interactions = []
        for target_user, interact in target_users.items():
            interact_dict_with_weight = dict_plus_weight(interact.__dict__, (source_user, target_user))
            source_user_interactions.append((target_user, interact_dict_with_weight))

        graph_data.append((source_user, dict(source_user_interactions)))

    graph_data = dict(graph_data)
    graph = nx.DiGraph(graph_data)
    return graph


def to_undirected_gaprh(graph: nx.DiGraph) -> nx.Graph:
    # Make G undirected.
    original_edges_weights = {(i, j): d["weight"] for i, j, d in graph.edges(data=True)}
    graph = nx.Graph(graph)
    for (i, j) in graph.edges():
        graph[i][j]['weight'] = original_edges_weights.get((i, j), 0) + original_edges_weights.get((j, i), 0)

    return graph


def draw_user_interactions_graph(graph: nx.Graph, op: str = None, ax: plt.Axes = None, use_weight=False):
    # draw graph with different edges weights:
    pos = nx.spring_layout(graph, seed=1919)
    node_colors = ['b' if user != op else 'r' for user in graph.nodes]   # draw op node with different color
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, ax=ax)

    nx.draw_networkx_labels(graph, pos)

    # draw edges with different width according to weight function
    for u, v, edge_data in graph.edges(data=True):
        weight = edge_data["weight"] if use_weight else 1
        if weight > 1:
            weight
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight, ax=ax)

    plt.show()







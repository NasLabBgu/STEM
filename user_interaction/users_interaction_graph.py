from typing import Dict, Union, Callable, List

from user_interaction.user_interaction_parser import UsersInteraction

import networkx as nx

from matplotlib import pyplot as plt

DEFAULT_WEIGHT = 1.0


def get_one(data: dict): return DEFAULT_WEIGHT


def build_users_interaction_graph(users_interactions: Dict[str, Dict[str, UsersInteraction]],
                                  weight_func: Callable[[dict], float] = None,
                                  op: str = None) -> nx.Graph:

    if weight_func is None:
        weight_func = get_one

    graph_data = {out_user: {in_user: d.__dict__ for in_user, d in in_users.items()} for out_user, in_users in users_interactions.items()}
    graph = nx.DiGraph(graph_data)

    # draw graph with different edges weights:
    pos = nx.spring_layout(graph, seed=1919)
    nx.draw_networkx_nodes(graph, pos, node_color='b')

    # draw op node with different color
    if op is not None:
        nx.draw_networkx_nodes(graph, pos, nodelist=[op], node_color='r')

    nx.draw_networkx_labels(graph, pos)

    # draw edges with different width according to weight function
    for u, v, edge_data in graph.edges(data=True):
        weight = weight_func(edge_data)
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight)

    plt.show()
    return graph






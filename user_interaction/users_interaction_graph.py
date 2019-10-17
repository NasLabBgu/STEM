from typing import Dict, Union, Callable, List

from user_interaction.user_interaction_parser import UsersInteraction

import networkx as nx

from matplotlib import pyplot as plt


def build_users_interaction_graph(users_interactions: Dict[str, Dict[str, UsersInteraction]],
                                  weights: Union[Callable, List] = None) -> nx.Graph:

    graph_data = {out_user: {in_user: d.__dict__ for in_user, d in in_users.items()} for out_user, in_users in users_interactions.items()}
    graph = nx.DiGraph(graph_data)
    data = graph.edges(data=True)
    print(data)
    # draw graph with different edges weights:
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    for u, v, edge_data in graph.edges(data=True):
        weight = edge_data["num_replies"]
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight)

    plt.show()
    return graph



from typing import Dict, Union, Callable, List

import pylab
# from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import graphviz_layout

from stance_classification.classifiers.maxcut import new_figure
from stance_classification.user_interaction.user_interaction_parser import UsersInteraction

import networkx as nx

from matplotlib import pyplot as plt

DEFAULT_WEIGHT = 1.0

OP_COLOR = 'green'
NODES_COLOR = 'lightblue'


def get_one(*args, **kwargs): return DEFAULT_WEIGHT


def build_users_interaction_graph(users_interactions: Dict[str, Dict[str, UsersInteraction]],
                                  weight_func: Callable[[dict], float] = None
                                  ) -> nx.DiGraph:

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


def draw_user_interactions_graph(graph: nx.Graph, op: str = None,
                                 ax: plt.Axes = None, use_weight: bool = False,
                                 outpath: str = None, title: str = None):
    fig = new_figure()
    # draw graph with different edges weights:
    # pos = nx.spring_layout(graph, seed=1919)
    pos = graphviz_layout(graph, prog='dot')
    node_colors = [NODES_COLOR if user != op else OP_COLOR for user in graph.nodes]
    nx.draw_networkx(graph, pos, node_color=node_colors)
    if use_weight:
        labels = {e: '{}'.format(graph[e[0]][e[1]]['weight']) for e in graph.edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    if title is not None:
        fig.suptitle(title, fontsize=16, y=0.97)

    # Show the figure.
    if outpath is not None:
        pylab.savefig(outpath)

    pylab.show()










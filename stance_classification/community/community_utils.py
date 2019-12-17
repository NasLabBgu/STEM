from typing import List, Iterable

import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import random
import numpy as np
from operator import itemgetter

from stance_classification.community.community_detection import edge_betweenness


def plot_betweeness(G, pos=None, source=None):

    if pos is None:
        pos = nx.spring_layout(G, seed=1919)
    # edge_labels = {k: str(v) for k,v in nx.algorithms.centrality.load._edge_betweenness(G, source='B').items()}
    if source is None:
        edge_labels = calc_betweeness(G)
    else:
        edge_labels = edge_betweenness(G, source)
    nx.draw_networkx(G, pos=pos, edge_labels=edge_labels)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    # betweenness = nx.edge_betweenness_centrality(G)
    # print(betweenness)

    plt.show()


def calc_betweeness(G, cutoff=False):
    betweenness = {}
    for u, v in G.edges():
        betweenness[(u, v)] = 0.0
        betweenness[(v, u)] = 0.0

    for source in G:
        ubetween = edge_betweenness(G, source, cutoff=cutoff)
        for e, ubetweenv in ubetween.items():
            betweenness[e] += ubetweenv / 2  # cumulative total

    return betweenness


def get_op_community(graph: nx.Graph, communities: List[List[str]], op: str):
    for community in communities:
        if op in community:
            return graph.subgraph(community)
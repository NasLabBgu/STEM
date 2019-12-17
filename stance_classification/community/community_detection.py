from typing import List, Iterable

import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import performance as modularity_score
import matplotlib.pyplot as plt
import random
import numpy as np
from operator import itemgetter


def modularity_score2(G: nx.Graph, communities: List[List[str]]):
    n_doubled_edges = 2 * len(G.edges)
    res = 0
    for c in communities:
        for i in c:
            for j in c:
                A_ij = int(j in G[i])
                expected = G.degree[i] * G.degree(j) * 1. / n_doubled_edges
                res += A_ij - expected

    return res / n_doubled_edges


def plot_partitions_score(G):
    for i in range(1):
        random.seed(246 + i)
        np.random.seed(4812 + i)
        communities_generator = community.girvan_newman(G, most_valuable_edge=find_most_central_edge)
        n_clusters = []
        scores = []
        for com in [[G.nodes]] + list(communities_generator):
            print(sorted(map(sorted, com)))
            print("num clusters: {}".format(len(com)))
            s = modularity_score(com, G)
            n_clusters.append(len(com))
            scores.append(s)
            print(s)
            print("\n")

        plt.plot(n_clusters, scores)
        print("===============================")

    plt.show()


def find_most_central_edge(graph: nx.Graph):
    # edge_betweeness_values = nx.edge_betweenness_centrality(graph, weight="weight")
    edge_betweeness_values = nx.edge_betweenness_centrality(graph)
    max_edge, max_value = max(edge_betweeness_values.items(), key=itemgetter(1))
    print(f"{max_edge}: {max_value}")
    return max_edge


def find_communities(graph: nx.Graph) -> Iterable[List[str]]:
    communities_generator = community.girvan_newman(graph, most_valuable_edge=find_most_central_edge)
    all_community_separations = [[graph.nodes]]
    scores = [modularity_score2(graph, [graph.nodes])]
    for communities in communities_generator:
        all_community_separations.append(communities)
        num_clusters = len(communities)
        s = modularity_score(graph, communities)
        scores.append(s)

        print(f"{num_clusters}: {s} -> {s / np.log2(np.log2(num_clusters + 1))}")

    # find max score and return the corresponding separation to communities
    max_index = np.argmax(map(lambda s, c: s / np.log2(np.log2(len(c)+1)), zip(scores, all_community_separations)))
    print(max_index)
    return all_community_separations[max_index]


def spatial_bipartition(G):
    nodes = sorted(G.nodes)
    laplace_repr = nx.laplacian_matrix(G, nodes)
    print("Laplacian Matrix Representation:\n", "  " + "  ".join(nodes), "\n", laplace_repr.todense())
    eigenvalues, eigenvectors = np.linalg.eig(laplace_repr.todense())
    index, second_smallest_val = sorted(enumerate(eigenvalues), key=lambda t: t[1])[1]
    second_smallest_vector = eigenvectors[:, index].A.ravel()
    print(second_smallest_val)
    print(second_smallest_vector)

    threshold = 0
    communities = [[], []]
    for i in range(len(nodes)):
        communitiy = second_smallest_vector[i] >= threshold
        communities[communitiy].append(nodes[i])

    print(communities)
    score = modularity_score(communities, G)
    print(score)


def edge_betweenness(G, source, nodes=None, cutoff=False):
    """Edge betweenness helper."""
    # get the predecessor data
    (pred, length) = nx.predecessor(G, source, cutoff=cutoff, return_seen=True)
    # order the nodes by path length
    onodes = [n for n, d in sorted(length.items(), key=itemgetter(1))]
    # initialize betweenness, doesn't account for any edge weights
    between = {}
    for u, v in G.edges(nodes):
        between[(u, v)] = 0.0
        between[(v, u)] = 0.0

    while onodes:  # work through all paths
        v = onodes.pop()
        if v in pred:
            # print()
            # Discount betweenness if more than one shortest path.
            num_paths = len(pred[v])
            # print(v, ": ", num_paths)
            for w in pred[v]:
                if w in pred:
                    # Discount betweenness, mult path
                    num_paths_w = len(pred[w])
                    between[(v, w)] += 1. / num_paths
                    between[(w, v)] += 1. / num_paths
                    # print("   ", w, ": ", num_paths_w)
                    for x in pred[w]:
                        # print("     ", x, ": ", between[(v, w)] / num_paths_w)
                        between[(w, x)] += between[(v, w)] / num_paths_w
                        between[(x, w)] += between[(w, v)] / num_paths_w
    return between


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


if __name__ == "__main__":
    # build graph:
    G = nx.Graph()
    G.add_nodes_from([chr(i) for i in range(65, 74)])
    G.add_edges_from(
        [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'H'), ('C', 'D'), ('D', 'F'), ('D', 'E'), ('E', 'F'), ('E', 'G'),
         ('G', 'I'), ('I', 'H'), ('H', 'G')])
    # el = calc_betweeness(G)
    # print(el)
    # nodes_positions = dict(
    #     I=(-1, -1), F=(1, -1), A=(0, 1),
    #     B=(-0.33, 0.33), C=(0.33, 0.33), H=(-0.66, -0.33),
    #     D=(0.66, -0.33), G=(-0.33, -1), E=(0.33, -1))
    # plot_betweeness(G, pos=nodes_positions)

    plot_partitions_score(G)
    spatial_bipartition(G)




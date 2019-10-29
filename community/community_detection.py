import networkx as nx
from networkx.algorithms import community
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt
import random
import numpy as np
from operator import itemgetter


def modularity_score(communities, G):
    n_doubled_edges = 2 * len(G.edges)
    res = 0
    for c in communities:
        for i in c:
            for j in c:
                # if i == j:
                # 	continue
                # print("nodes: ({}, {})".format(i, j))
                A_ij = int(j in G[i])
                # print("A_ij: {}".format(A_ij))
                expected = G.degree[i] * G.degree(j) * 1. / n_doubled_edges
                # print("{} * {} / 2m = {}".format(G.degree[i], G.degree[j], expected))
                res += A_ij - expected
            # print("{} - {} = {}".format(A_ij, expected, A_ij - expected))
            # print()

    return res / n_doubled_edges


def find_best_partition(G):
    for i in range(1):
        random.seed(246 + i)
        np.random.seed(4812 + i)
        communities_generator = community.girvan_newman(G)
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

    find_best_partition(G)
    spatial_bipartition(G)




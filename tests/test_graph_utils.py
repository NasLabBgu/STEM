from unittest import TestCase

import networkx as nx

from stance_classification.graph_utils import count_all_triplets, calc_global_clustering_coefficient


class TestGraphUtils(TestCase):

    def test_count_all_triplets(self):
        g1 = get_graph1()
        expected = 6
        actual = count_all_triplets(g1)
        self.assertEqual(expected, actual)

        g2 = get_graph2()
        expected = 7
        actual = count_all_triplets(g2)
        self.assertEqual(expected, actual)

        g3 = get_clique(5)
        expected = 30
        actual = count_all_triplets(g3)
        self.assertEqual(expected, actual)

    def test_calc_global_clustering_coefficient(self):
        g1 = get_graph1()
        expected = 0
        actual = calc_global_clustering_coefficient(g1)
        self.assertEqual(expected, actual)

        g2 = get_graph2()
        expected = 3. / 7
        actual = calc_global_clustering_coefficient(g2)
        self.assertEqual(expected, actual)

        g3 = get_clique(5)
        expected = 1
        actual = calc_global_clustering_coefficient(g3)
        self.assertEqual(expected, actual)


def get_graph1():
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return nx.Graph(edges).to_undirected()


def get_graph2() -> nx.Graph:
    edges = [(0, 1), (1, 2), (2, 3), (3, 1), (3, 4)]
    return nx.Graph(edges).to_undirected()


def get_clique(n: int) -> nx.Graph:
    edges = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
    return nx.Graph(edges).to_undirected()


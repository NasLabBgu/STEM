from operator import itemgetter
from typing import Dict


import networkx as nx

from stance_classification.user_interaction.user_interaction_parser import UsersInteraction


def remove_unconnected_interactions():
    pass


def is_single_user(user_name: str, user_interactions: Dict[str, UsersInteraction]):
    if len(user_interactions) == 0:
        return True

    if len(user_interactions) == 1:
        if user_name in user_interactions:
            return True

    return False


def remove_nodes_without_interactions(interactions: Dict[str, Dict[str, UsersInteraction]]) -> Dict[str, Dict[str, UsersInteraction]]:
    num_users = len(interactions)
    new_interactions = {user: interact for user, interact in interactions.items() if not is_single_user(user, interact)}
    while len(new_interactions) < num_users:
        num_users = len(new_interactions)
        new_interactions = {user: interact for user, interact in interactions.items() if
                            not is_single_user(user, interact)}


def remove_nodes_with_low_interactions(graph: nx.Graph, degree_threshold) -> nx.Graph:
    high_degree_nodes = [k for k, d in graph.degree if d >= degree_threshold]
    return graph.subgraph(high_degree_nodes)


def get_op_connected_component(graph: nx.Graph, op: str) -> set:
    return nx.node_connected_component(graph, op)


def get_all_pairs(n: int):
    return (n * (n-1)) / 2


def count_all_triplets(graph: nx.Graph) -> int:
    num_triplets = 0
    for v in graph.nodes:
        num_neighbors = sum(1 for _ in graph.neighbors(v))
        num_triplets_around_node = get_all_pairs(num_neighbors)
        num_triplets += num_triplets_around_node

    return num_triplets


def calc_global_clustering_coefficient(graph: nx.Graph) -> float:
    num_triangles = sum(nx.triangles(graph).values())
    num_triplets = count_all_triplets(graph)
    return float(num_triangles) / num_triplets


def inter_communication_index(graph: nx.Graph) -> float:
    avg_deg = float(sum(map(itemgetter(1), graph.degree))) / graph.number_of_nodes()
    triplets_dominance = float(count_all_triplets(graph)) / graph.number_of_nodes()
    # high_degree_ratio = float(len(high_degree_nodes)) / graph.number_of_nodes()
    return triplets_dominance * avg_deg



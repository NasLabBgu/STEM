from collections import Counter
from operator import itemgetter
from typing import Dict, List

import networkx as nx

from stance_classification.user_interaction.user_interaction_parser import UsersInteraction

# tree dict fields
from treetools.TreeTools import walk_tree

NODE_FIELD = "node"
AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"
LABELS_FIELD = "labels"


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


def get_node_name(counter: Counter, author: str, anonimous_index: dict = None) -> str:
    if anonimous_index is not None:
        user_index = anonimous_index.setdefault(author, len(anonimous_index))
        author = f"user{user_index}"

    counter[author] += 1
    return f"{author}-{counter[author]}"


def tree_to_graph(tree: dict, anonimous: bool = False) -> nx.Graph:
    first_node = tree[NODE_FIELD]
    op: str = first_node[AUTHOR_FIELD]
    authors_counts = Counter()
    anonimous_index = {} if anonimous else None
    current_branch_authors: List[str] = []

    node_name = get_node_name(authors_counts, op, anonimous_index)

    tree_graph = nx.OrderedDiGraph()
    tree_graph.add_node(node_name, **first_node)
    current_branch_authors.append(node_name)

    tree_nodes = walk_tree(tree)
    next(tree_nodes)  # skip the first node
    for depth, node in tree_nodes:
        # check if the entire current branch was parsed, and start walking to the next branch
        if depth < len(current_branch_authors):
            del current_branch_authors[depth:]

        current_author = node[AUTHOR_FIELD]
        node_name = get_node_name(authors_counts, current_author, anonimous_index)
        tree_graph.add_node(node_name, **node)

        prev_author = current_branch_authors[-1]
        tree_graph.add_edge(prev_author, node_name)

        current_branch_authors.append(node_name)

    return tree_graph



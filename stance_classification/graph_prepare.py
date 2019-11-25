from typing import Tuple

import networkx as nx

from data_analyze import filter_interactions
from gold_standard.create_gold_labels import extract_labeled_pairs
from graph_utils import get_op_connected_component
from user_interaction.user_interaction_parser import parse_users_interactions
from user_interaction.users_interaction_graph import build_users_interaction_graph, to_undirected_gaprh


def preprocess_from_tree(tree: dict, remove_irrelevant_interactions: bool = True) -> nx.Graph:
    interactions = parse_users_interactions(tree)
    op = tree["node"]["author"]
    if remove_irrelevant_interactions:
        interactions = filter_interactions(interactions, op)

    def calculate_edge_weight(edge_data: dict, edge: Tuple[str, str] = None) -> float:
        return edge_data["num_replies"] + edge_data["num_quotes"] + (
                edge_data["num_confirmed_delta_awards"] + edge_data["num_rejected_delta_awards"]) * 3

    graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight)
    undir_graph = to_undirected_gaprh(graph)
    op_connected_nodes = get_op_connected_component(undir_graph, op)
    undir_graph = undir_graph.subgraph(op_connected_nodes)
    return undir_graph

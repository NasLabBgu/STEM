from typing import Dict

from user_interaction.user_interaction_parser import UsersInteraction

import networkx as nx


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


def get_op_connected_component(graph: nx.Graph, op: str) -> nx.Graph:
    return nx.node_connected_component(graph, op)


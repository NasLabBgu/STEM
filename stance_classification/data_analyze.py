
import argparse
import json

from typing import List, Iterable, Dict, Tuple, Callable, Set

import numpy as np
import pandas as pd

from stance_classification.classifiers.maxcut import max_cut, draw_maxcut
from stance_classification.user_interaction.user_interaction_parser import parse_users_interactions, UsersInteraction
from stance_classification.user_interaction.users_interaction_graph import build_users_interaction_graph, draw_user_interactions_graph, \
    to_undirected_gaprh
from stance_classification.utils import iter_trees_from_jsonl


IRRELEVANT_USERS = {None, "DeltaBot", "[deleted]"}


def is_relevant_user(source_user: str, op: str, source_users: Set[str], target_users: Set[str]) -> bool:
    if source_user in IRRELEVANT_USERS:
        return False

    irrelevant_users = IRRELEVANT_USERS | {source_user}
    filtered_targets = target_users - irrelevant_users
    filtered_sources = source_users - irrelevant_users
    if len(filtered_targets) == 0:
        return False

    if len(filtered_sources) == 0:
        return False

    # check if the user replied only to the op and the op didn't replied back
    if len(filtered_targets) == 1:
        if (op in target_users) and (op not in source_users):
            return False

    return True


def filter_interactions(users_interactions: Dict[str, Dict[str, dict]], op: str):

    # build reversed interactions
    reversed_interactions = {user: set() for user in users_interactions.keys()}
    for source_user, interactions in users_interactions.items():
        for target_user in interactions.keys():
            sources = reversed_interactions.setdefault(target_user, set())
            sources.add(source_user)

    filtered_interactions = []
    for out_user, interactions in users_interactions.items():
        target_users = interactions.keys()
        source_users = reversed_interactions[out_user]

        if is_relevant_user(out_user, op, source_users, target_users):

            filtered_user_interactions = \
                {in_user: d for in_user, d in interactions.items()
                       if in_user not in IRRELEVANT_USERS and in_user != out_user
                 }

            filtered_interactions.append((out_user, filtered_user_interactions))

    return dict(filtered_interactions)


def analyze_data(trees: Iterable[dict]):

    for i, tree in enumerate(trees):
        op = tree["node"]["author"]
        print(f"Tree: {i} ;\tOP: {op}")
        interactions = parse_users_interactions(tree)
        print(json.dumps(tree, indent=4))

        def calculate_edge_weight(edge_data: dict, edge: Tuple[str, str]) -> float:
            return edge_data["num_replies"] + edge_data["num_quotes"] + (edge_data["num_confirmed_delta_awards"] + edge_data["num_rejected_delta_awards"]) * 3

        interactions = filter_interactions(interactions, op, min_op_interact=2, weight_func=calculate_edge_weight)
        graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight)

        # draw_user_interactions_graph(graph, op=op, use_weight=True, outpath=f"/home/ron/workspace/bgu/stance-classification/examples/users-interactions-{i}-latest.png")

        if graph.number_of_nodes() <= 1:
            continue

        undir_graph = to_undirected_gaprh(graph)
        rval, cut_nodes = max_cut(undir_graph)
        draw_maxcut(undir_graph, cut_nodes, rval, op, outpath=f"/home/ron/workspace/bgu/stance-classification/examples/users-interactions-{i}-maxcut-latest.png")


        # print(json.dumps(interactions, indent=4, default=lambda cls: cls.__dict__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="trees in json format to analyze")

    # args = parser.parse_args()
    args = parser.parse_args(["/home/ron/data/bgu/trees_2.0.txt"])

    trees = iter_trees_from_jsonl(args.data)
    analyze_data(trees)



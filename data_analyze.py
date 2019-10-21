
import argparse
import json

from typing import List, Iterable, Dict, Tuple

import numpy as np
import pandas as pd

from stance_classification.maxcut import max_cut, draw_maxcut
from user_interaction.user_interaction_parser import parse_users_interactions
from user_interaction.users_interaction_graph import build_users_interaction_graph, draw_user_interactions_graph, \
    to_undirected_gaprh
from utils import iter_trees_from_jsonl


IRRELEVANT_USERS = {None, "DeltaBot", "[deleted]"}


def remove_irrelevant_users_interaction(users_interactions: Dict[str, Dict[str, dict]]):
    global IRRELEVANT_USERS
    users_interactions = {
        out_user: {in_user: d
                   for in_user, d in in_users.items()
                   if in_user not in IRRELEVANT_USERS
                   }
        for out_user, in_users in users_interactions.items()
        if out_user not in IRRELEVANT_USERS
    }
    return users_interactions


def analyze_data(trees: Iterable[dict]):

    for i, tree in enumerate(trees):
        print(f"Tree: {i}")
        interactions = parse_users_interactions(tree)
        # interactions = remove_irrelevant_users_interaction(interactions)
        print(json.dumps(tree, indent=4))
        op = tree["node"]["author"]

        def calculate_edge_weight(edge_data: dict, edge: Tuple[str, str]) -> float:
            return edge_data["num_replies"] + edge_data["num_quotes"]

        graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight)

        draw_user_interactions_graph(graph, op=op, use_weight=False)
        draw_user_interactions_graph(graph, op=op, use_weight=True)

        if graph.number_of_nodes() <= 1:
            continue

        undir_graph = to_undirected_gaprh(graph)
        rval, cut_nodes = max_cut(undir_graph)
        draw_maxcut(undir_graph, cut_nodes, rval, op)


        # print(json.dumps(interactions, indent=4, default=lambda cls: cls.__dict__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="trees in json format to analyze")

    # args = parser.parse_args()
    args = parser.parse_args(["/home/ron/data/bgu/trees_2.0.txt"])

    trees = iter_trees_from_jsonl(args.data)
    analyze_data(trees)



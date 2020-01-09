
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



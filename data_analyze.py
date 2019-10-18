
import argparse
import json

from typing import List, Iterable

import numpy as np
import pandas as pd

from stance_classification.maxcut import max_cut
from user_interaction.user_interaction_parser import parse_users_interactions
from user_interaction.users_interaction_graph import build_users_interaction_graph
from utils import iter_trees_from_jsonl


def calculate_edge_weight(edge_data: dict) -> float:
    return edge_data["num_replies"]


def analyze_data(trees: Iterable[dict]):

    for i, tree in enumerate(trees):
        print(f"Tree: {i}")
        interactions = parse_users_interactions(tree)
        print(json.dumps(tree, indent=4))
        op = tree["node"]["author"]
        graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight, op=op)

        weights = {(i, j): calculate_edge_weight(d) for i, j, d in graph.edges(data=True)}

        max_cut(graph, weights)

        # print(json.dumps(interactions, indent=4, default=lambda cls: cls.__dict__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="trees in json format to analyze")

    # args = parser.parse_args()
    args = parser.parse_args(["/home/ron/data/bgu/trees_2.0.txt"])

    trees = iter_trees_from_jsonl(args.data)
    analyze_data(trees)



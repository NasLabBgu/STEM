import json
from operator import itemgetter
from typing import Iterable, Tuple, List, Dict

import networkx as nx

from stance_classification.community.community_detection import plot_partitions_score, spatial_bipartition, find_communities
from stance_classification.community.community_utils import get_op_community
from stance_classification.draw_utils import draw_tree
from stance_classification.graph_utils import inter_communication_index, tree_to_graph
from stance_classification.classifiers.greedy_stance_classifier import MSTStanceClassifier
from stance_classification.classifiers.maxcut import max_cut, draw_maxcut
from stance_classification.classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from stance_classification.classifiers.random_stance_classifier import RandomStanceClassifier
from stance_classification.user_interaction.user_interaction_parser import parse_users_interactions, UsersInteraction, \
    cmv_interaction_weight
from stance_classification.user_interaction.users_interaction_graph import build_users_interaction_graph, \
    draw_user_interactions_graph, \
    to_undirected_gaprh, get_core_interactions
from stance_classification.utils import iter_trees_from_jsonl, skip_elements

DISAGREEMENT_TAGS = {"CBE", "CBD", "CBF", "OCQ", "DNO"}   #CBB - ad hominem


def extract_labeled_pairs(interactions: Dict[str, Dict[str, UsersInteraction]]) -> Tuple[str, str, dict]:
    labeled_pairs = list(filter(itemgetter(2), ((user1, user2, pair_interact.labels)
                         for user1, interact in interactions.items()
                            for user2, pair_interact in interact.items())))

    return labeled_pairs


def get_certain_labels(pair_labels: List[Dict[str, List[str]]]) -> List[str]:
    agreed_labels = set()

    if len(pair_labels) == 0:
        return None

    for annotation in pair_labels:
        # annotation is a dict with annotator name as key and its annotation as value
        annotators_answers = list(annotation.values())
        agreed_answers = set(annotators_answers[0]).intersection(*annotators_answers)
        agreed_labels.update(agreed_answers)

    return agreed_labels


def show_labels(trees: Iterable[dict]):

    num_skip = 0
    skip_elements(trees, num_skip)
    for i, tree in enumerate(trees, num_skip):
        op = tree["node"]["author"]
        if op == "[deleted]":
            continue

        tree_graph = tree_to_graph(tree)
        draw_tree(tree_graph, path="/home/ron/workspace/bgu/stance-classification/plots/tree.png")

        print(f"Tree: {i} ; OP: {op} ; Title: {tree['node']['extra_data']['title']}")
        interactions = parse_users_interactions(tree)
        pairs_labels = extract_labeled_pairs(interactions)
        for user1, user2, all_pair_labels in pairs_labels:
            if len(pairs_labels) > 0:
                certain_pair_labels = get_certain_labels(all_pair_labels)
                # print(f"{user1} --> {user2} : {certain_pair_labels}")

        # remove_irrelevant_interactions = True
        show_digraph = False
        show_graph = True
        compare_classifiers = False
        compute_communities = False

        undir_graph = get_core_interactions(interactions, op)
        if undir_graph.number_of_nodes() == 0:
            continue

        inter_communication_score = inter_communication_index(undir_graph)
        title = f"{tree['node']['id']}" \
                f"\n{tree['node']['extra_data']['title']}" \
                f"\ninter-comm-index: {inter_communication_score}" \
                f"\nnum nodes: {undir_graph.number_of_nodes()}"
        # title = tree['node']['title']

        if show_digraph:
            print(f"number of nodes: {graph.number_of_nodes()}")
            print(f"number of edges: {graph.number_of_edges()}")
            draw_user_interactions_graph(graph, op=op, use_weight=True, title=title)

        if show_graph:
            draw_user_interactions_graph(undir_graph, op=op, use_weight=True, title=title)

        if compare_classifiers:
            rsc = RandomStanceClassifier()
            rsc.set_input(undir_graph)
            rsc.classify_stance(op, 1./3)
            rsc.draw()

            msc = MSTStanceClassifier()
            msc.set_input(undir_graph)
            msc.classify_stance(op)
            msc.draw()

            maxcut_clf = MaxcutStanceClassifier()
            maxcut_clf.set_input(undir_graph)
            try:
                maxcut_clf.classify_stance(op)
                maxcut_clf.draw()
            except OverflowError as e:
                print(e)

        if compute_communities:
            communities = find_communities(graph)
            graph = get_op_community(graph, communities, op)
            # plot_betweeness(graph)
            # find_best_partition(graph)
            draw_user_interactions_graph(graph, op=op, use_weight=True)
            undir_graph = to_undirected_gaprh(graph)
            spatial_bipartition(undir_graph)



if __name__ == "__main__":

    # labeled_trees_path = "/home/ron/data/bgu/labeled/labeled_trees.jsonl"
    labeled_trees_path = "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    trees = iter_trees_from_jsonl(labeled_trees_path)
    show_labels(trees)


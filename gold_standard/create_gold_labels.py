import json
from operator import itemgetter
from typing import Iterable, Tuple, List, Dict

from community.community_detection import plot_partitions_score, spatial_bipartition, find_communities
from community.community_utils import get_op_community
from data_analyze import filter_interactions
from graph_utils import remove_nodes_without_interactions, get_op_connected_component
from stance_classification.greedy_stance_classifier import MSTStanceClassifier
from stance_classification.maxcut import max_cut, draw_maxcut
from stance_classification.maxcut_stance_classifier import MaxcutStanceClassifier
from stance_classification.random_stance_classifier import RandomStanceClassifier
from user_interaction.user_interaction_parser import parse_users_interactions, UsersInteraction
from user_interaction.users_interaction_graph import build_users_interaction_graph, draw_user_interactions_graph, \
    to_undirected_gaprh
from utils import iter_trees_from_jsonl, skip_elements

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

    num_skip = 15
    skip_elements(trees, num_skip)
    for i, tree in enumerate(trees, num_skip):
        op = tree["node"]["author"]
        if op == "[deleted]":
            continue

        print(f"Tree: {i} ; OP: {op} ; Title: {tree['node']['extra_data']['title']}")
        interactions = parse_users_interactions(tree)
        pairs_labels = extract_labeled_pairs(interactions)
        for user1, user2, all_pair_labels in pairs_labels:
            if len(pairs_labels) > 0:
                certain_pair_labels = get_certain_labels(all_pair_labels)
                # print(f"{user1} --> {user2} : {certain_pair_labels}")

        # show graph
        def calculate_edge_weight(edge_data: dict, edge: Tuple[str, str]=None) -> float:
            return edge_data["num_replies"] + edge_data["num_quotes"] + (edge_data["num_confirmed_delta_awards"] + edge_data["num_rejected_delta_awards"]) * 3

        remove_irrelevant_interactions = True
        show_digraph = False
        show_graph = True
        compare_classifiers = True
        compute_communities = False

        if remove_irrelevant_interactions:
            interactions = filter_interactions(interactions, op)
            # interactions = remove_nodes_without_interactions(interactions)

        graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight)
        undir_graph = to_undirected_gaprh(graph)
        op_connected_nodes = get_op_connected_component(undir_graph, op)
        # graph = graph.subgraph(op_connected_nodes)
        undir_graph = undir_graph.subgraph(op_connected_nodes)

        title = f"{tree['node']['id']}\n{tree['node']['extra_data']['title']}"
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


import json
from operator import itemgetter
from typing import Iterable, Tuple, List, Dict

from community.community_detection import plot_partitions_score, spatial_bipartition, find_communities
from community.community_utils import get_op_community
from data_analyze import remove_irrelevant_users_interaction
from graph_utils import remove_nodes_without_interactions, get_op_connected_component
from stance_classification.maxcut import max_cut, draw_maxcut
from stance_classification.random_stance_classifier import RandomStanceClassifier
from user_interaction.user_interaction_parser import parse_users_interactions, UsersInteraction
from user_interaction.users_interaction_graph import build_users_interaction_graph, draw_user_interactions_graph, \
    to_undirected_gaprh
from utils import iter_trees_from_jsonl

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

    for i, tree in enumerate(trees):
        op = tree["node"]["author"]
        print(f"Tree: {i} ; OP: {op}")
        interactions = parse_users_interactions(tree)
        pairs_labels = extract_labeled_pairs(interactions)
        for user1, user2, all_pair_labels in pairs_labels:
            if len(pairs_labels) > 0:
                certain_pair_labels = get_certain_labels(all_pair_labels)
                print(f"{user1} --> {user2} : {certain_pair_labels}")

        # show graph
        def calculate_edge_weight(edge_data: dict, edge: Tuple[str, str]=None) -> float:
            return edge_data["num_replies"] + edge_data["num_quotes"] + (edge_data["num_confirmed_delta_awards"] + edge_data["num_rejected_delta_awards"]) * 3

        interactions = remove_irrelevant_users_interaction(interactions, op, min_op_interact=2,
                                                           weight_func=calculate_edge_weight)

        # interactions = remove_nodes_without_interactions(interactions)

        graph = build_users_interaction_graph(interactions, weight_func=calculate_edge_weight)
        undir_graph = to_undirected_gaprh(graph)
        op_connected_nodes = get_op_connected_component(undir_graph, op)
        # graph = graph.subgraph(op_connected_nodes)
        undir_graph = undir_graph.subgraph(op_connected_nodes)

        rsc = RandomStanceClassifier()
        rsc.set_input(undir_graph)
        rsc.classify_stance(op, 0.5)
        rsc.draw()

        title = tree['node']["extra_data"]['title']
        # title = tree['node']['title']
        draw_user_interactions_graph(undir_graph, op=op, use_weight=True, title=title)

        # communities = find_communities(graph)
        # graph = get_op_community(graph, communities, op)
        # # plot_betweeness(graph)
        # # find_best_partition(graph)
        # draw_user_interactions_graph(graph, op=op, use_weight=True)
        # undir_graph = to_undirected_gaprh(graph)
        # spatial_bipartition(undir_graph)
        try:
            rval, cut_nodes = max_cut(undir_graph)
            draw_maxcut(undir_graph, cut_nodes, rval, op)
        except OverflowError as e:
            print(e)
            continue


if __name__ == "__main__":

    # labeled_trees_path = "/home/ron/data/bgu/labeled/labeled_trees.jsonl"
    labeled_trees_path = "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    trees = iter_trees_from_jsonl(labeled_trees_path)
    show_labels(trees)


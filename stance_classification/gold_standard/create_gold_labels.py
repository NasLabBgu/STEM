import json
from operator import itemgetter
from typing import Iterable, Tuple, List, Dict

import networkx as nx


from stance_classification.draw_utils import draw_tree
from stance_classification.graph_utils import inter_communication_index, tree_to_graph
from stance_classification.classifiers.greedy_stance_classifier import MSTStanceClassifier
from stance_classification.classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from stance_classification.classifiers.random_stance_classifier import RandomStanceClassifier
from stance_classification.user_interaction.users_interaction_graph import draw_user_interactions_graph, \
     get_core_interactions
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

    ## Flags
    # remove_irrelevant_interactions = True
    anonimous = False
    parse_tree = False
    show_digraph = False
    show_graph = False
    compare_classifiers = True
    compute_communities = False

    num_skip = 0
    skip_elements(trees, num_skip)
    for i, tree in enumerate(trees, num_skip):
        op = tree["node"]["author"]
        if op == "[deleted]":
            continue

        # print(f"Tree: {i} ; OP: {op} ; Title: {tree['node']['extra_data']['title']}")
        interactions = parse_users_interactions(tree, anonimous=anonimous)
        pairs_labels = extract_labeled_pairs(interactions)
        for user1, user2, all_pair_labels in pairs_labels:
            if len(pairs_labels) > 0:
                certain_pair_labels = get_certain_labels(all_pair_labels)
                # print(f"{user1} --> {user2} : {certain_pair_labels}")

        if anonimous:
            op = "user0"

        if parse_tree:
            undir_graph = get_core_interactions(interactions, op, k_core=1)
            core_nodes = nx.k_core(undir_graph, 2).nodes
            if len(core_nodes) == 0:
                continue

            tree_graph = tree_to_graph(tree, anonimous=False)
            tree_nodes = [n for n in tree_graph.nodes if n.split("-")[0] in core_nodes]
            sub_tree_graph = tree_graph.subgraph(tree_nodes)
            sub_tree_graph = nx.k_core(sub_tree_graph, 1)

            op_root_name = f"{op}-1"
            relevant_nodes = [comp for comp in nx.connected_components(sub_tree_graph.to_undirected()) if op_root_name in comp][0]
            sub_tree_graph = sub_tree_graph.subgraph(relevant_nodes)
            tree_title = tree['node']['extra_data']['title']
            try:
                draw_tree(sub_tree_graph,
                      path=f"/home/<user>/workspace/bgu/stance-classification/plots/tree-{i}-limited-users.png",
                          title=tree_title)
            except:
                pass

        else:
            undir_graph = get_core_interactions(interactions, op, k_core=1)
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
            if show_graph:
                rsc.draw()

            msc = MSTStanceClassifier()
            msc.set_input(undir_graph)
            msc.classify_stance(op)
            if show_graph:
                msc.draw()

            maxcut_clf = MaxcutStanceClassifier()
            maxcut_clf.set_input(undir_graph)
            try:
                maxcut_clf.classify_stance(op)
                trees_index[len(trees_index)] = tree["node"]["id"]
                print(f"{len(trees_index)},{tree['node']['id']}")
                if show_graph:
                    maxcut_clf.draw()


            except OverflowError as e:
                print(e)




if __name__ == "__main__":
    trees_index = {}
    # labeled_trees_path = "/home/<user>/data/bgu/labeled/labeled_trees.jsonl"
    labeled_trees_path = "/home/<user>/data/bgu/labeled/61019_notcut_trees.txt"
    trees = iter_trees_from_jsonl(labeled_trees_path)
    show_labels(trees)


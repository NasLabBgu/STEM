from typing import Iterable, Tuple, Sequence, Dict

import numpy as np
import networkx as nx
from sklearn.metrics import classification_report

from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.graph_prepare import preprocess_from_tree
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from classifiers.random_stance_classifier import RandomStanceClassifier
from classifiers.stance_classification_utils import load_labels
from utils import iter_trees_from_jsonl, skip_elements


def get_confusion_matrix(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_true_neg = np.logical_not(y_true)
    y_pred_neg = np.logical_not(y_pred)

    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(y_true_neg, y_pred))
    fn = np.sum(np.logical_and(y_true, y_pred_neg))
    tn = np.sum(np.logical_and(y_true_neg, y_pred_neg))

    return tp, fp, fn, tn


def iter_graphs(trees: Iterable[dict], start: int = 0, end: int = -1) -> Iterable[Tuple[str, nx.Graph]]:
    skip_elements(trees, start)
    for i, tree in enumerate(trees, start):
        if i == end:
            break

        tree_id = tree['node']['id']
        op = tree["node"]["author"]
        if op == "[deleted]":
            continue

        print(f"Tree: {i} ; ID: {tree_id} ; OP: {op} ; Title: {tree['node']['extra_data']['title']}")
        undir_graph = preprocess_from_tree(tree)
        yield tree_id, op, undir_graph


def evaluate_classifier(stance_clf: BaseStanceClassifier, nodes_labels_map: Dict[Tuple[str, str], bool], trees: Iterable[dict]):
    labeled_nodes = nodes_labels_map.keys()
    labels = np.asarray(list(nodes_labels_map.values()))

    predictions = {}
    for tree_id, op, graph in iter_graphs(trees, end=18):
        stance_clf.set_input(graph)
        stance_clf.classify_stance(op)
        predictions.update({(tree_id, supporter): True for supporter in stance_clf.get_supporters()})
        predictions.update({(tree_id, opposer): False for opposer in stance_clf.get_complement()})

    # align labels
    # print("\n".join(map(str, predictions.keys())))
    y_pred = np.asarray([predictions[node] for node in labeled_nodes])
    print(classification_report(labels, y_pred))


if __name__ == "__main__":

    # labeled_trees_path = "/home/ron/data/bgu/labeled/labeled_trees.jsonl"
    labeled_trees_path = "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    labels_mapping = load_labels("/home/ron/data/bgu/labeled/stance_gs.csv")

    print("Evaluating Max Cut")
    trees = iter_trees_from_jsonl(labeled_trees_path)
    maxcut_clf = MaxcutStanceClassifier()
    evaluate_classifier(maxcut_clf, labels_mapping, trees)

    print("Evaluating MST (greedy approach)")
    trees = iter_trees_from_jsonl(labeled_trees_path)
    mstclf = MSTStanceClassifier()
    evaluate_classifier(mstclf, labels_mapping, trees)

    print("Evaluating Random")
    trees = iter_trees_from_jsonl(labeled_trees_path)
    prior = float(sum(labels_mapping.values())) / len(labels_mapping)
    print(f"labels prior: {prior}")
    mstclf = RandomStanceClassifier(random_seed=1919, support_prior=prior)
    evaluate_classifier(mstclf, labels_mapping, trees)



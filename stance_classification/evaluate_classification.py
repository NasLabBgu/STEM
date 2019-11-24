from typing import Iterable, Tuple, Sequence, Dict

import numpy as np

from stance_classification.base_stance_classifier import BaseStanceClassifier


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


def evaluate_classifier(stance_classifier: BaseStanceClassifier, nodes_labels_map: Dict[Tuple[str, str], bool]):
    labeled_nodes = nodes_labels_map.keys()
    labels = np.asarray(list(nodes_labels_map.values()))
    clf




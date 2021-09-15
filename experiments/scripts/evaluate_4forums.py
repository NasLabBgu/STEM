import time

from typing import List, Callable, Dict, Iterable, Any, Tuple, Sequence, Set, NamedTuple, Union

import os
import argparse
from itertools import groupby, starmap, chain
from operator import itemgetter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from conversant.conversation import Conversation

from sklearn.metrics import accuracy_score

from data.iac.fourforum_data import \
    load_post_records as load_4forums_post_records,\
    build_conversations as build_4forums_conversations
from data.iac.fourforum_labels import load_author_labels as load_4forums_author_labels
from experiments.datahandlers.iac.fourforum_interactions import FourForumInteractionsBuilder
from experiments.datahandlers.iac.fourforum_labels import AuthorLabel
from interactions import InteractionsGraph
from interactions.interactions_graph import PairInteractionsData

FOURFORUMS_DIR = "fourforums"
FOURFORUMS_AUTHOR_LABELS_FILENAME = "mturk_author_stance.txt"

# """abortion = 3
#        evolution = 7
#        gay marriage = 8
#        gun control = 9
#        """
RELEVANT_TOPICS = {3, 7, 8, 9}


# type aliases
ConversationsLoader = Callable[[str], List[Conversation]]
LabelsByConversation = Dict[Any, Dict[Any, int]]
PostLabels = Dict[Any, Dict[Any, int]]

class IncrementableInt:
    def __init__(self, init_value: int = 0):
        self.value = init_value

    def increment(self):
        self.value += 1

    def __repr__(self):
        return repr(self.value)

    def __str__(self):
        return str(self.value)


class TimeMeasure:
    def __init__(self):
        self.__start = None
        self.__end = None

    def start(self):
        if self.__start is None:
            self.__start = time.time()

        return self.__start

    def end(self):
        if self.__end is None:
            self.__end = time.time()

        return self.__end

    def duration(self):
        if self.__start is None or self.__end is None:
            raise AssertionError("Time not taken yet")

        return self.__end - self.__start


# LOAD CONVERSATIONS

def load_4forums_conversations(data_dir: str) -> List[Conversation]:
    records = tqdm(load_4forums_post_records(data_dir))
    return list(build_4forums_conversations(records))


loaders: Dict[str, ConversationsLoader] = {
    "4forums": load_4forums_conversations
}


def load_conversations(dataset_name: str, basedir: str) -> List[Conversation]:
    loader = loaders.get(dataset_name, None)
    if loader is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}. must be one of f{list(loaders.keys())}")

    return load_4forums_conversations(basedir)


# LOADER AND INFER LABELS


def create_author_labels_dict(labels: Iterable[AuthorLabel]) -> Dict[Any, int]:
    return {l.author_id: l.stance for l in labels if l.stance is not None}


def infer_posts_labels_from_authors(convs: List[Conversation], author_labels_per_conversation: LabelsByConversation) -> PostLabels:
    post_labels = {}
    for c in convs:
        cid = c.id
        authors_labels = author_labels_per_conversation.get(cid, None)
        if authors_labels is None:
            continue

        conv_post_labels = {node.node_id: authors_labels.get(node.author) for _, node in c.iter_conversation()}
        post_labels.update({(cid, k): v for k, v in conv_post_labels.items() if v is not None})

    return post_labels


def get_4forums_labels(data_dir: str) -> Tuple[LabelsByConversation, PostLabels]:
    author_labels_path = os.path.join(data_dir, FOURFORUMS_AUTHOR_LABELS_FILENAME)
    author_labels = list(load_4forums_author_labels(author_labels_path))

    author_labels_per_conversation = groupby(author_labels, key=lambda a: a.discussion_id)
    author_labels_per_conversation = starmap(lambda cid, labels: (cid, create_author_labels_dict(labels)), author_labels_per_conversation)
    author_labels_per_conversation = filter(lambda cid_to_labels: len(cid_to_labels[1]) > 0, author_labels_per_conversation)
    author_labels_per_conversation = dict(author_labels_per_conversation)

    post_labels_per_conversation = infer_posts_labels_from_authors(convs, author_labels_per_conversation)
    return author_labels_per_conversation, post_labels_per_conversation


# GRAPH UTILITIES
def get_ordered_candidates_for_pivot(graph: nx.Graph, weight_field: str = "weight") -> Sequence[Any]:
    inv_weight_field = "inv_weight"
    for _, _, pair_data in graph.edges(data=True):
        weight = pair_data.data[weight_field]
        pair_data.data[inv_weight_field] = 1 / weight

    node_centralities = nx.closeness_centrality(graph, distance=inv_weight_field)
    return list(map(itemgetter(0), sorted(node_centralities.items(), key=itemgetter(1), reverse=True)))


def get_pivot_node(graph: nx.Graph, labeled_authors: Union[Set[Any], Dict[Any, Any]], weight_field: str = "weight") -> Any:
    candidates = get_ordered_candidates_for_pivot(graph, weight_field=weight_field)
    return next(iter(filter(labeled_authors.__contains__, candidates)), None)


# EVALUATION UTILITIES
def extend_preds(graph: nx.Graph, seed_node: Any, core_authors_preds: Dict[Any, int]) -> Dict[Any, int]:
    extended_results = dict(core_authors_preds.items())
    for (n1, n2) in nx.bfs_edges(graph, source=seed_node):
        if n2 not in extended_results:
            n1_label = extended_results[n1]
            extended_results[n2] = 1 - n1_label

    return extended_results


class EvaluationUtils:
    def __init__(self, author_labels_per_conversation: LabelsByConversation, post_labels: PostLabels):
        self.author_labels_per_conversation = author_labels_per_conversation
        self.post_labels = post_labels

    def get_authors_labels_in_conv(self, conv: Conversation) -> Dict[Any, int]:
        if conv.id not in self.author_labels_per_conversation:
            return None

        return self.author_labels_per_conversation[conv.id]

    def get_author_preds(self, clf: BaseStanceClassifier, pivot: Any, authors_labels: Dict[Any, int] = None, conv: Conversation = None) -> Dict[Any, int]:
        if authors_labels is None:
            if conv is None:
                raise ValueError("At least one of 'author_labels' or 'conv' must be not None")

            authors_labels = self.get_authors_labels_in_conv(conv)

        support_label = authors_labels[pivot]
        opposer_label = 1 - support_label
        supporters = clf.get_supporters()
        opposers = clf.get_complement()
        preds = {}
        for supporter in supporters:
            preds[supporter] = support_label
        for opposer in opposers:
            preds[opposer] = opposer_label

        return preds


def get_maxcut_results(graph: InteractionsGraph, op: Any) -> MaxcutStanceClassifier:
    maxcut = MaxcutStanceClassifier(weight_field=graph.WEIGHT_FIELD)
    maxcut.set_input(graph.graph, op)
    maxcut.classify_stance()
    return maxcut


def get_greedy_results(graph: InteractionsGraph, op: Any) -> BaseStanceClassifier:
    clf = MSTStanceClassifier() #weight_field=graph.WEIGHT_FIELD)
    clf.set_input(graph.graph)
    clf.classify_stance(op)
    return clf


def align_gs_with_predictions(authors_labels: Dict[Any, int], author_preds: Dict[Any, int]) -> Tuple[List[int], List[int]]:
    y_true, y_pred = [], []
    for author, true_label in authors_labels.items():
        pred = author_preds.get(author, None)
        if pred is None: continue

        y_true.append(true_label)
        y_pred.append(pred)

    return y_true, y_pred


def predict_for_partition(true: List[int], preds: List[int]) -> Tuple[List[int], List[int]]:
    acc = accuracy_score(true, preds)
    if acc < 0.5:
        preds = [1-l for l in preds]

    return true, preds


def get_best_preds(true_labels: Dict[Any, int], pred_labels: Dict[Any, int]) -> Dict[Any, int]:
    true, preds = align_gs_with_predictions(true_labels, pred_labels)
    acc = accuracy_score(true, preds)
    if acc < 0.5:
        return {k: (1-  l) for k, l in pred_labels.items()}

    return pred_labels


def get_posts_preds(conv: Conversation, post_labels: Dict[Any, int], author_preds: Dict[Any, int]) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    posts_true, posts_pred = {}, {}
    conv_id = conv.id
    for depth, node in conv.iter_conversation():
        label = post_labels.get((conv_id, node.node_id), None)
        if label is None: continue
        pred = author_preds.get(node.author, None)
        if pred is None: continue

        posts_true[node.node_id] = label
        posts_pred[node.node_id] = pred

    return posts_true, posts_pred


def calc_weight(interactions: PairInteractionsData) -> float:
    n_replies = interactions["replies"]
    n_quotes = interactions["quotes"]
    return (0.02 * n_replies) + n_quotes


class ExperimentResults(NamedTuple):
    total_count = IncrementableInt()
    on_topic_count = IncrementableInt()
    total_time = TimeMeasure()
    convs_by_id: Dict[Any, Conversation] = {}
    full_graphs: Dict[Any, InteractionsGraph] = {}
    core_graphs: Dict[Any, InteractionsGraph] = {}
    maxcut_results: Dict[Any, MaxcutStanceClassifier] = {}
    pivot_nodes: Dict[Any, Any] = {}
    author_predictions: Dict[Any, Dict[str, Dict[Any, int]]] = {}
    posts_predictions: Dict[Any, Dict[str, Dict[Any, int]]] = {}
    empty_core: List[int] = []
    unlabeled_conversations: List[int] = []
    unlabeled_op: List[int] = []
    insufficient_author_labels: List[int] = []
    too_small_cut_value: List[int] = []
    op_not_in_core: List[int] = []
    large_graphs: List[int] = []
    single_author_conv: List[int] = []


def process_stance(conversations: Sequence[Conversation], evalutils: EvaluationUtils, naive_results: bool = False, ) -> ExperimentResults:
    interactions_parser = FourForumInteractionsBuilder()
    results = ExperimentResults()
    print("Start processing authors stance")
    results.total_time.start()
    for i, conv in tqdm(enumerate(conversations), total=len(conversations)):
        results.total_count.increment()
        topic = conv.root.data["topic"]
        if topic not in RELEVANT_TOPICS:
            continue

        results.on_topic_count.increment()
        authors_labels = evalutils.get_authors_labels_in_conv(conv)
        if authors_labels is None:
            results.unlabeled_conversations.append(i)
            continue

        if len(authors_labels) == 0:
            results.insufficient_author_labels.append(i)
            continue

        interaction_graph = interactions_parser.build(conv)
        interaction_graph.set_interaction_weights(calc_weight)
        zero_edges = [(v, u) for v, u, d in interaction_graph.graph.edges(data=True) if d["weight"] == 0]
        interaction_graph.graph.remove_edges_from(zero_edges)

        if len(conv.participants) <= 1:
            results.single_author_conv.append(i)
            continue

        results.convs_by_id[conv.id] = conv
        results.full_graphs[conv.id] = interaction_graph

        pivot_node = get_pivot_node(interaction_graph.graph, authors_labels, weight_field="weight")
        results.pivot_nodes[conv.id] = pivot_node

        mst = get_greedy_results(interaction_graph, pivot_node)
        preds = evalutils.get_author_preds(mst, pivot_node, authors_labels=authors_labels)
        results.author_predictions[conv.id] = {"mst": preds}

        if naive_results:
            continue

        core_interactions = interaction_graph.get_core_interactions()
        results.core_graphs[conv.id] = core_interactions
        if core_interactions.graph.size() == 0:
            results.empty_core.append(i)
            continue

        components = list(nx.connected_components(core_interactions.graph))
        core_interactions = core_interactions.get_subgraph(components[0])
        pivot_node = get_pivot_node(core_interactions.graph, authors_labels, weight_field="weight")
        maxcut = get_maxcut_results(core_interactions, pivot_node)
        if maxcut.cut_value < 3:
            results.too_small_cut_value.append(i)
            continue

        results.maxcut_results[conv.id] = maxcut

        # if core_interactions.graph.order() > 120:
        #     large_graphs.append(conv)
        #     continue

        preds = evalutils.get_author_preds(maxcut, pivot_node, authors_labels=authors_labels)
        results.author_predictions[conv.id]["core"] = preds

        # get extended results
        preds = extend_preds(interaction_graph.graph, pivot_node, preds)
        results.author_predictions[conv.id]["full"] = preds

    results.total_time.end()
    return results


def show_results(results: ExperimentResults):
    print(f"total time took: {results.total_time.duration()}")
    print(f"total number of conversations (in all topics): {len(convs)}")
    print(f"total number of conversations (in the relevant topics): {results.on_topic_count}")
    print(f"total number of conversations with labeled authors (in all topics): {len(author_labels_per_conversation)}")
    print(
        f"total number of conversations with labeled authors (in the relevant topics): {results.on_topic_count.value - len(results.unlabeled_conversations)}")
    print(f"number of conversations in eval: {len(results.convs_by_id)}")
    labeled_authors = sum(len(v) for v in author_labels_per_conversation.values())
    print(f"total number of labeled authors: {labeled_authors}")
    print(f"number of conversations in eval: {len(results.convs_by_id)}")
    all_authors_in_eval = set(
        chain(*[predictions["mst"].keys() for cid, predictions in results.author_predictions.items()]))
    print(f"number of unique authors in eval: {len(all_authors_in_eval)}")
    all_authors_in_core_eval = set(
        chain(*[predictions.get("core", {}).keys() for cid, predictions in results.author_predictions.items()]))
    print(f"number of unique authors in core: {len(all_authors_in_core_eval)}")
    print("=========")
    print(f"number of conversations with single author: {len(results.single_author_conv)}")
    print(f"number of conversations with empty core: {len(results.empty_core)}")
    print(f"number of conversations with op not in core: {len(results.op_not_in_core)}")
    print(f"number of conversations with too large core: {len(results.large_graphs)}")
    print(f"number of conversations with too small cut value: {len(results.too_small_cut_value)}")
    print(f"number of unlabeled conversations: {len(results.unlabeled_conversations)}")
    print(f"number of conversations with unlabeled op: {len(results.unlabeled_op)}")
    print(f"number of conversations with insufficient labeled authors: {len(results.insufficient_author_labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")

    args = parser.parse_args()

    convs = load_conversations(args.dataset, args.path)
    author_labels_per_conversation, posts_labels = get_4forums_labels(args.path)
    evalutils = EvaluationUtils(author_labels_per_conversation, posts_labels)
    results = process_stance(convs, evalutils)
    show_results(results)




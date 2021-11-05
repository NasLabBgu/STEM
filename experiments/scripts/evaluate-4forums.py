import time

from typing import List, Dict, Iterable, Any, Tuple, Sequence, Set, NamedTuple, Union

import os
import argparse
from itertools import groupby, starmap, chain, islice
from operator import itemgetter

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from conversant.conversation import Conversation

from sklearn.metrics import accuracy_score, recall_score

from conversation.parse import DataFrameConversationReader
from experiments.datahandlers.iac.fourforum_interactions import FourForumInteractionsBuilder
from experiments.utils import IncrementableInt, TimeMeasure
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
MST_MODEL = "MST"
CORE_MODEL = "2-CORE"
FULL_MODEL = "FULL"

MODELS = [MST_MODEL, CORE_MODEL, FULL_MODEL]


# type aliases

LabelsByConversation = Dict[Any, Dict[Any, int]]
PostLabels = Dict[Any, Dict[Any, int]]

# LOADING CONVERSATIONS
fields_mapping = {
    "node_id": "post_id",
    "author": "author",
    "timestamp": "timestamp",
    "parent_id": "parent_post_id"
}


def load_conversations_from_dataframe(path: str) -> Iterable[Conversation]:
    df = pd.read_csv(path)
    parser = DataFrameConversationReader(fields_mapping, conversation_id_column="conversation_id")
    groups = df.groupby("conversation_id")
    for cid, raw_conversation in tqdm(groups, total=groups.ngroups):
        yield parser.parse(raw_conversation, conversation_id=cid)


def get_labels(conv: Conversation, label_field: str) -> Dict[Any, int]:
    labels = {}
    for depth, node in conv.iter_conversation():
        label = node.data[label_field]
        if label >= 0:
            labels[node.author] = label

    return labels


def get_author_labels(conv: Conversation) -> Dict[Any, int]:
    return get_labels(conv, "author_label")


def get_posts_labels(conv: Conversation) -> Dict[Any, int]:
    return get_labels(conv, "post_label")


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


def get_author_preds(clf: BaseStanceClassifier, pivot: Any, authors_labels: Dict[Any, int]) -> Dict[Any, int]:
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


def align_gs_with_predictions(true_labels: Dict[Any, int], preds: Dict[Any, int]) -> Tuple[List[int], List[int]]:
    y_true, y_pred = [], []
    for author, true_label in true_labels.items():
        pred = preds.get(author, None)
        if pred is None:
            continue

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
        if label is None:
            continue

        pred = author_preds.get(node.author, None)
        if pred is None:
            continue

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

    def show(self):
        print(f"total time took: {self.total_time.duration()}")
        print(f"total number of conversations (in all topics): {self.total_count}")
        print(f"total number of conversations (in the relevant topics): {self.on_topic_count}")
        print(
            f"total number of conversations with labeled authors (in the relevant topics): {self.on_topic_count.value - len(self.unlabeled_conversations)}")
        print(f"number of conversations in eval: {len(self.convs_by_id)}")
        all_authors_in_eval = set(
            chain(*[predictions[MST_MODEL].keys() for cid, predictions in self.author_predictions.items()]))
        print(f"number of unique authors in eval: {len(all_authors_in_eval)}")
        all_authors_in_core_eval = set(
            chain(*[predictions.get(CORE_MODEL, {}).keys() for cid, predictions in self.author_predictions.items()]))
        print(f"number of unique authors in core: {len(all_authors_in_core_eval)}")
        print("=========")
        print(f"number of conversations with single author: {len(self.single_author_conv)}")
        print(f"number of conversations with empty core: {len(self.empty_core)}")
        print(f"number of conversations with op not in core: {len(self.op_not_in_core)}")
        print(f"number of conversations with too large core: {len(self.large_graphs)}")
        print(f"number of conversations with too small cut value: {len(self.too_small_cut_value)}")
        print(f"number of unlabeled conversations: {len(self.unlabeled_conversations)}")
        print(f"number of conversations with unlabeled op: {len(self.unlabeled_op)}")
        print(f"number of conversations with insufficient labeled authors: {len(self.insufficient_author_labels)}")


def process_stance(
        conversations: Iterable[Conversation],
        # author_labels_per_conversation: LabelsByConversation,
        naive_results: bool = False
) -> ExperimentResults:
    interactions_parser = FourForumInteractionsBuilder()
    results = ExperimentResults()
    print("Start processing authors stance")
    results.total_time.start()
    for i, conv in enumerate(conversations):
        results.total_count.increment()
        topic = conv.root.data["topic_name"]
        # if topic not in RELEVANT_TOPICS:
        #     continue

        results.on_topic_count.increment()
        authors_labels = get_author_labels(conv)
        if len(authors_labels) is None:
            results.unlabeled_conversations.append(i)
            continue

        if len(authors_labels) == 0:
            results.insufficient_author_labels.append(i)
            continue

        interaction_graph = interactions_parser.build(conv)
        interaction_graph.set_interaction_weights(calc_weight)
        zero_edges = [(v, u) for v, u, d in interaction_graph.graph.edges(data=True) if d["weight"] == 0]
        interaction_graph.graph.remove_edges_from(zero_edges)

        if conv.number_of_participants <= 1:
            results.single_author_conv.append(i)
            continue

        results.convs_by_id[conv.id] = conv
        results.full_graphs[conv.id] = interaction_graph

        pivot_node = get_pivot_node(interaction_graph.graph, authors_labels, weight_field="weight")
        results.pivot_nodes[conv.id] = pivot_node

        mst = get_greedy_results(interaction_graph, pivot_node)
        preds = get_author_preds(mst, pivot_node, authors_labels=authors_labels)
        results.author_predictions[conv.id] = {MST_MODEL: preds}

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

        preds = get_author_preds(maxcut, pivot_node, authors_labels=authors_labels)
        results.author_predictions[conv.id][CORE_MODEL] = preds

        # get extended results
        preds = extend_preds(interaction_graph.graph, pivot_node, preds)
        results.author_predictions[conv.id][FULL_MODEL] = preds

    results.total_time.end()
    return results


def results_to_df(results: ExperimentResults) -> pd.DataFrame:
    records = []
    for conv_id, predictions in results.author_predictions.items():
        conv = results.convs_by_id[conv_id]
        n_authors = conv.number_of_participants
        topic = conv.root.data["topic_name"]
        author_labels = get_author_labels(conv)
        all_posts_labels = get_posts_labels(conv)

        for model in MODELS:
            author_preds = predictions.get(model, None)
            if author_preds is None:
                continue

            author_preds_best = get_best_preds(author_labels, author_preds)
            posts_preds, posts_preds = get_posts_preds(conv, all_posts_labels, author_preds)
            posts_preds_best = get_best_preds(posts_preds, posts_preds)
            for _, node in conv.iter_conversation():
                author_label = author_labels.get(node.author)
                post_label = all_posts_labels.get(node.node_id)
                if (author_label or post_label) is None:
                    continue

                record = {"node_id": node.node_id, "author": node.author, "conv_id": conv_id, "topic": topic,
                          "authors": n_authors, "posts": conv.size, "model": model,
                          "author_label": author_label, "author_pred": author_preds.get(node.author), "author_pred_best": author_preds_best.get(node.author),
                          "post_label": post_label, "post_pred": posts_preds.get(node.node_id), "post_pred_best": posts_preds_best.get(node.node_id)}

                records.append(record)

    return pd.DataFrame.from_records(records)


def get_metrics(labels: Dict[Any, int], preds: Dict[Any, int], suffix: str = None, best_group_label_assignment: bool = False) -> dict:
    if suffix is None:
        suffix = ""
    else:
        suffix = f"-{suffix}"

    if best_group_label_assignment:
        preds = get_best_preds(labels, preds)

    y_true, y_pred = align_gs_with_predictions(labels, preds)
    return {
        f"support": len(y_true),
        f"accuracy{suffix}": accuracy_score(y_true, y_pred),
        f"recall-1{suffix}": recall_score(y_true, y_pred, pos_label=1),
        f"recall-0{suffix}": recall_score(y_true, y_pred, pos_label=0)
    }


def eval_results(results: ExperimentResults) -> pd.DataFrame:
    records = []
    for conv_id, predictions in results.author_predictions.items():
        conv = results.convs_by_id[conv_id]
        n_authors = conv.number_of_participants
        topic = conv.root.data["topic_name"]
        author_labels = get_author_labels(conv)
        all_posts_labels = get_posts_labels(conv)

        for model in MODELS:
            author_preds = predictions.get(model, None)
            if author_preds is None:
                continue

            record = {"conv_id": conv_id, "topic": topic, "authors": n_authors, "posts": conv.size, "model": model}

            # author level metrics
            record.update(get_metrics(author_labels, author_preds, suffix="author"))
            record.update(get_metrics(author_labels, author_preds, suffix="author-best", best_group_label_assignment=True))

            # posts level metrics
            posts_labels, posts_preds = get_posts_preds(conv, all_posts_labels, author_preds)
            record.update(get_metrics(posts_labels, posts_preds, suffix="post"))
            record.update(get_metrics(posts_labels, posts_preds, suffix="post-best", best_group_label_assignment=True))

            records.append(record)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")

    args = parser.parse_args()

    convs = load_conversations_from_dataframe(args.path)
    results = process_stance(convs)
    results.show()

    results_df = results_to_df(results)
    results_df.to_csv("4forums-maxcut-results.csv")

    eval_df = eval_results(results)
    eval_df.to_csv("4forums-eval.csv")
    print(eval_df)




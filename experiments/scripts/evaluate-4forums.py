import time
from collections import Counter
from functools import partial

from typing import List, Dict, Iterable, Any, Tuple, Sequence, Set, NamedTuple, Union, Callable, Optional

import os
import argparse
from itertools import groupby, starmap, chain, islice
from operator import itemgetter
import heapq

import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm

from classifiers.base_stance_classifier import BaseStanceClassifier
from classifiers.greedy_stance_classifier import MSTStanceClassifier
from classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from conversant.conversation import Conversation

from sklearn.metrics import accuracy_score, recall_score, precision_score

from conversation.parse import DataFrameConversationReader
from experiments.datahandlers.iac.fourforum_interactions import FourForumInteractionsBuilder
from experiments.utils import IncrementableInt, TimeMeasure
from interactions import InteractionsGraph
from interactions.interactions_graph import PairInteractionsData

FOURFORUMS_DIR = "fourforums"
FOURFORUMS_AUTHOR_LABELS_FILENAME = "mturk_author_stance.txt"


POSITIVE_STANCE_LABEL: int = 1
NEGATIVE_STANCE_LABEL: int = 0


POSITIVE_PRED_INDEX: int = 1
NEGATIVE_PRED_INDEX: int = 0

GREEDY_MODEL = "GREEDY"
STEM_CORE_MODEL = "STEM-CORE"
STEM_PRPG_MODEL = "STEM-PROPAGATED"
MCSN_CORE_MODEL = "STEM+ZS+SN-CORE"
MCSN_PRPG_MODEL = "STEM+ZS+SN-PROPAGATED"
MCSN_MODEL_ALL = "STEM+ZS+SN-ALL"
STEM_PRPG_ZSSD = 'STEM+ZSSD'
ZSSD_AVG = "ZS-AVG"
ZSSD_MAX = "ZS-MAX"
ZSSD_SUM = "ZS-SUM"


MODELS = [GREEDY_MODEL, STEM_CORE_MODEL, STEM_PRPG_MODEL, STEM_PRPG_ZSSD, MCSN_CORE_MODEL, MCSN_PRPG_MODEL, MCSN_MODEL_ALL, ZSSD_AVG, ZSSD_MAX, ZSSD_SUM]
NEGATIVE_STANCE_NODE = "stance-N"
POSITIVE_STANCE_NODE = "stance-P"
STANCE_EDGE_WEIGHT = 1.0

# type aliases

LabelsByConversation = Dict[Any, Dict[Any, int]]
PostLabels = Dict[Any, Dict[Any, int]]

# LOADING CONVERSATIONS
FIELDS_MAPPING = {
    "node_id": "post_id",
    "author": "author",
    "timestamp": "timestamp",
    "parent_id": "parent_post_id"
}


def empty_zs_preds() -> np.ndarray:
    return np.zeros(3)


EMPTY_ZS_PREDS = empty_zs_preds()


def load_conversations_from_dataframe(path: str) -> Iterable[Conversation]:
    df = pd.read_csv(path, low_memory=False)
    parser = DataFrameConversationReader(FIELDS_MAPPING, conversation_id_column="conversation_id")
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
    labels = {}
    for depth, node in conv.iter_conversation():
        label = node.data["author_label"]
        if isinstance(label, str):
            print(node)
        if label >= 0:
            labels[node.author] = label

    return labels

#4forums_10847_5


def get_posts_labels(conv: Conversation) -> Dict[Any, int]:
    labels = {}
    for depth, node in conv.iter_conversation():
        label = node.data["post_label"]
        if label >= 0:
            labels[node.node_id] = label

    return labels


def get_zs_post_labels(path: str, probs: bool = True) -> Dict[Any, Union[int, np.ndarray]]:
    df = pd.read_csv(path)
    if probs:
        return dict(zip(df["ori_id"], df[["pred-0", "pred-1", "pred-2"]].values))

    return dict(zip(df["ori_id"], df["pred"]))


# GRAPH UTILITIES
def get_ordered_candidates_for_pivot(graph: nx.Graph, weight_field: str = "weight") -> Sequence[Any]:
    inv_weight_field = "inv_weight"
    for _, _, pair_data in graph.edges(data=True):
        weight = pair_data.data[weight_field]
        pair_data.data[inv_weight_field] = 1 / weight

    node_centralities = nx.closeness_centrality(graph, distance=inv_weight_field)
    return list(map(itemgetter(0), sorted(node_centralities.items(), key=itemgetter(1), reverse=True)))


def get_pivot_node(graph: nx.Graph, op: Any, weight_field: str = "weight") -> Any:
    if op in graph:
        return op

    candidates = get_ordered_candidates_for_pivot(graph, weight_field=weight_field)
    return candidates[0]


# EVALUATION UTILITIES
def extend_preds(graph: nx.Graph, seed_node: Any, core_authors_preds: Dict[Any, int]) -> Dict[Any, int]:
    extended_results = dict(core_authors_preds.items())
    for (n1, n2) in nx.bfs_edges(graph, source=seed_node):
        if n2 not in extended_results:
            n1_label = extended_results[n1]
            extended_results[n2] = 1 - n1_label

    return extended_results


def get_author_preds(positive_authors: Set[Any], negative_authors: Set[Any]) -> Dict[Any, int]:
    preds = {}
    for author in positive_authors:
        preds[author] = POSITIVE_STANCE_LABEL
    for author in negative_authors:
        preds[author] = NEGATIVE_STANCE_LABEL

    return preds


def get_maxcut_results(graph: InteractionsGraph, op: Any) -> MaxcutStanceClassifier:
    maxcut = MaxcutStanceClassifier(weight_field=graph.WEIGHT_FIELD)
    maxcut.set_input(graph.graph, op)
    maxcut.classify_stance()
    return maxcut


def get_maxcut_with_stance_results(graph: nx.Graph, op: Any, weight_field: str = "weight") -> MaxcutStanceClassifier:
    maxcut = MaxcutStanceClassifier(weight_field=weight_field, stance_nodes=(POSITIVE_STANCE_NODE, NEGATIVE_STANCE_NODE))
    maxcut.set_input(graph, op)
    maxcut.classify_stance()
    return maxcut


def decide_stance_groups_by_stance_nodes(supporters: Set[Any], opposers: Set[Any]) -> Tuple[Set[Any], Set[Any]]:
    if NEGATIVE_STANCE_NODE in opposers:
        assert POSITIVE_STANCE_NODE in supporters
        return opposers, supporters

    assert NEGATIVE_STANCE_NODE in supporters
    assert POSITIVE_STANCE_NODE in opposers
    return supporters, opposers


def get_greedy_results(graph: InteractionsGraph, op: Any) -> BaseStanceClassifier:
    clf = MSTStanceClassifier()  # weight_field=graph.WEIGHT_FIELD)
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
        preds = [1 - l for l in preds]

    return true, preds


def get_best_preds(true_labels: Dict[Any, int], pred_labels: Dict[Any, int]) -> Dict[Any, int]:
    true, preds = align_gs_with_predictions(true_labels, pred_labels)
    acc = accuracy_score(true, preds)
    if acc < 0.5:
        return {key: (1 - label) for key, label in pred_labels.items()}

    return pred_labels


def decide_stance_groups_by_zs(conv: Conversation, supporters: Set[Any], opposers: Set[Any],
                               zs_labels: Dict[Any, int], strategy: str = "sum-neg") -> Tuple[Set[Any], Set[Any]]:
    """
    decide which group has a positive stance, and each group has a negative stance.
    return the two groups, where the negative group first, followed by the positive group
    """
    supported_posts = list(map(lambda n: n[1].node_id, filter(lambda n: n[1].author in supporters, conv.iter_conversation())))
    opposed_posts = list(map(lambda n: n[1].node_id, filter(lambda n: n[1].author in opposers, conv.iter_conversation())))

    if strategy == "average":
        supporters_stance_sum = sum(map(zs_labels.get, supported_posts)) / len(supported_posts)
        opposers_stance_sum = sum(map(zs_labels.get, opposed_posts)) / len(opposed_posts)
    elif strategy == "sum-neg":
        supporters_stance_sum = sum(map(lambda x: (2 * x) - 1, map(zs_labels.get, supported_posts)))
        opposers_stance_sum = sum(map(lambda x: (2 * x) - 1, map(zs_labels.get, opposed_posts)))
    else:
        raise ValueError(f"No such strategy: {strategy}")

    if supporters_stance_sum > opposers_stance_sum:
        return opposers, supporters

    return supporters, opposers


def get_posts_preds(conv: Conversation, post_labels: Dict[Any, int], author_preds: Dict[Any, int]) -> Tuple[
    Dict[Any, int], Dict[Any, int]]:
    posts_true, posts_pred = {}, {}
    for depth, node in conv.iter_conversation():
        label = post_labels.get(node.node_id)
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
    return (0.5 * n_replies) + n_quotes


class ExperimentResults(NamedTuple):
    total_count = IncrementableInt()
    on_topic_count = IncrementableInt()
    total_time = TimeMeasure()
    convs_by_id: Dict[Any, Conversation] = {}
    full_graphs: Dict[Any, InteractionsGraph] = {}
    core_graphs: Dict[Any, InteractionsGraph] = {}
    maxcut_results: Dict[Any, MaxcutStanceClassifier] = {}
    maxcut_with_stance_results: Dict[Any, MaxcutStanceClassifier] = {}
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
    confident_convs: Set[Any] = set()
    confident_core_convs: Set[Any] = set()

    def show(self):
        print(f"total time took: {self.total_time.duration()}")
        print(f"total number of conversations (in all topics): {self.total_count}")
        print(f"total number of conversations (in the relevant topics): {self.on_topic_count}")
        print(
            f"total number of conversations with labeled authors (in the relevant topics): {self.on_topic_count.value - len(self.unlabeled_conversations)}")
        print(f"number of conversations in eval: {len(self.convs_by_id)}")
        all_authors_in_eval = set(
            chain(*[predictions[GREEDY_MODEL].keys() for cid, predictions in self.author_predictions.items()]))
        print(f"number of unique authors in eval: {len(all_authors_in_eval)}")
        all_authors_in_core_eval = set(
            chain(*[predictions.get(STEM_CORE_MODEL, {}).keys() for cid, predictions in self.author_predictions.items()]))
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


def is_significant_interactions(graph: InteractionsGraph) -> bool:
    # check that the structure is complex
    if not nx.is_k_regular(graph.graph, 2):
        return True

    return False

    # if graph.graph.size() >= 5:
    #     return True
    #
    # # check that not all nteractions have the same weight
    # interactions_weights = (i.interactions["weight"] for i in graph.interactions)
    # return len(set(interactions_weights)) > 1


def softmax(values: Sequence[float]) -> np.ndarray:
    if sum(values) == 0:
        return np.asarray(values)

    values_exp = np.exp(values)
    return values_exp / sum(values_exp)


def aggregate_authors_posts_preds(conv: Conversation, zs_labels: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
    authors_agg_preds = {}
    for _, node in conv.iter_conversation():
        normalized_pred = softmax(zs_labels.get(node.node_id, EMPTY_ZS_PREDS))
        authors_agg_preds.setdefault(node.author, []).append(normalized_pred)

    # convert list of arrays to matrix, and return
    return {author: np.vstack(preds) for author, preds in authors_agg_preds.items()}


def author_preds_from_zs_labels(author_posts_preds: Dict[Any, np.ndarray], strategy: str = "avg") -> Dict[Any, int]:
    agg_func: Callable[[np.ndarray], np.ndarray]
    if strategy == "avg":
        agg_func = partial(np.average, axis=0)
    elif strategy == "sum":
        agg_func = partial(np.sum, axis=0)
    elif strategy == "max":
        agg_func = partial(np.max, axis=0)

    authors_preds = {author: np.argmax(agg_func(preds)[:2]) for author, preds in author_posts_preds.items()}
    return authors_preds


def sum_top_k(values: Iterable[float], k) -> float:
    return sum(heapq.nlargest(k, values))


def aggregate_author_stance_preds(preds: np.ndarray) -> Dict[int, float]:
    """
    describe what's going on
    """
    max_pred_indices = np.argmax(preds, axis=1)
    max_pred_values = np.max(preds, axis=1)
    max_index_value_pairs = zip(max_pred_indices, max_pred_values)
    max_index_value_pairs = filter(lambda p: p[1] > 0.7, max_index_value_pairs)
    grouped_pairs_by_index = groupby(max_index_value_pairs, key=itemgetter(0))
    return {max_index: sum_top_k(map(itemgetter(1), pairs), 3) / 3 for max_index, pairs in grouped_pairs_by_index}


def connect_stance_nodes(interactions: InteractionsGraph, authors_agg_preds: Dict[Any, np.ndarray], weight: float = 1.0) -> nx.Graph:
    weighted_edges = [(i.user1, i.user2, {"weight": i.data[i.WEIGHT_FIELD]}) for i in interactions.interactions]
    total_weight = sum(map(itemgetter("weight"), map(itemgetter(2), weighted_edges)))

    existing_authors = interactions.graph.nodes
    authors_avg_preds = {author: aggregate_author_stance_preds(preds) for author, preds in authors_agg_preds.items() if author in existing_authors}
    total_stance_edges_weight = sum(map(lambda x: x.get(POSITIVE_PRED_INDEX, 0.0) + x.get(NEGATIVE_PRED_INDEX, 0.0), authors_avg_preds.values()))
    stance_weight_factor = weight * (total_weight / total_stance_edges_weight)

    pos_stance_edges = [(author, NEGATIVE_STANCE_NODE, {"weight": stance_weight_factor * preds.get(POSITIVE_PRED_INDEX, 0.0)}) for author, preds in authors_avg_preds.items()]
    neg_stance_edges = [(author, POSITIVE_STANCE_NODE, {"weight": stance_weight_factor * preds.get(NEGATIVE_PRED_INDEX, 0.0)}) for author, preds in authors_avg_preds.items()]

    # total_stance_edges_weight = 2. * len(authors_agg_preds)
    constraint_edge = (POSITIVE_STANCE_NODE, NEGATIVE_STANCE_NODE, {"weight": total_weight})

    return nx.from_edgelist(weighted_edges + pos_stance_edges + neg_stance_edges + [constraint_edge])
    # return nx.from_edgelist(weighted_edges + pos_stance_edges + neg_stance_edges)
    # return nx.from_edgelist(weighted_edges + stance_edges + [constraint_edge])


def are_confident_labels(authors_preds: Dict[Any, np.ndarray], confidence_threshold: float = 0.7,
                         min_confident_authors: int = 1, authors: Optional[Set[Any]] = None) -> bool:

    confident_authors_count = 0
    for author, posts_preds in authors_preds.items():
        if authors and author not in authors:
            continue

        if np.any(posts_preds > confidence_threshold):
            confident_authors_count += 1
            if confident_authors_count >= min_confident_authors:
                return True

    return False


def process_stance(
        conversations: Iterable[Conversation],
        decide_stance_groups: Callable[[Conversation, Set[Any], Set[Any]], Tuple[Set[Any], Set[Any]]],
        zs_preds: Dict[Any, np.ndarray],
        naive_results: bool = False,
) -> ExperimentResults:
    interactions_parser = FourForumInteractionsBuilder()
    results = ExperimentResults()
    print("Start processing authors stance")
    results.total_time.start()
    for i, conv in enumerate(pbar := tqdm(conversations)):
        results.total_count.increment()

        if conv.number_of_participants <= 1:
            results.single_author_conv.append(i)
            continue

        interaction_graph = interactions_parser.build(conv)
        interaction_graph.set_interaction_weights(calc_weight)
        zero_edges = [(v, u) for v, u, d in interaction_graph.graph.edges(data=True) if d["weight"] == 0]
        interaction_graph.graph.remove_edges_from(zero_edges)

        results.convs_by_id[conv.id] = conv
        results.full_graphs[conv.id] = interaction_graph

        pivot = get_pivot_node(interaction_graph.graph, conv.root.node_id)
        mst = get_greedy_results(interaction_graph, pivot)
        negative, positive = decide_stance_groups(conv, mst.get_supporters(), mst.get_complement())
        preds = get_author_preds(negative, positive)
        results.author_predictions[conv.id] = {GREEDY_MODEL: preds}

        # store zero-shot preds
        author_posts_preds = aggregate_authors_posts_preds(conv, zs_preds)

        if are_confident_labels(author_posts_preds, min_confident_authors=3):
            results.confident_convs.add(conv.id)

        for (model, strategy) in [(ZSSD_SUM, "sum"), (ZSSD_MAX, "max"), (ZSSD_AVG, "avg")]:
            author_preds = author_preds_from_zs_labels(author_posts_preds)
            results.author_predictions[conv.id][model] = author_preds

        if naive_results:
            continue

        stance_interactions_graph = connect_stance_nodes(interaction_graph, author_posts_preds, STANCE_EDGE_WEIGHT)
        pbar.set_description(f"Conversation {conv.id}, participants: {conv.number_of_participants}")
        maxcut_with_stance = get_maxcut_with_stance_results(stance_interactions_graph, conv.root.node_id, "weight")
        results.maxcut_with_stance_results[conv.id] = maxcut_with_stance
        # maxcut_with_stance.draw(outpath=f"graphs/{MCSN_MODEL_ALL}-{conv.id}.png")

        negative, positive = decide_stance_groups_by_stance_nodes(maxcut_with_stance.get_supporters(),
                                                                  maxcut_with_stance.get_complement())
        preds = get_author_preds(positive, negative)
        results.author_predictions[conv.id][MCSN_MODEL_ALL] = preds

        core_interactions = interaction_graph.get_core_interactions()
        results.core_graphs[conv.id] = core_interactions
        if core_interactions.graph.size() == 0:
            results.empty_core.append(i)
            continue

        if not is_significant_interactions(core_interactions):
            results.too_small_cut_value.append(i)
            continue

        components = list(nx.connected_components(core_interactions.graph))
        core_interactions = core_interactions.get_subgraph(components[0])

        if are_confident_labels(author_posts_preds, min_confident_authors=3, authors=set(core_interactions.graph.nodes)):
            results.confident_core_convs.add(conv.id)

        pivot = get_pivot_node(core_interactions.graph, conv.root.node_id)
        maxcut = get_maxcut_results(core_interactions, pivot)
        results.maxcut_results[conv.id] = maxcut

        negative, positive = decide_stance_groups(conv, maxcut.get_supporters(), maxcut.get_complement())
        preds = get_author_preds(positive, negative)
        results.author_predictions[conv.id][STEM_CORE_MODEL] = preds

        # propagate results from core to full graph
        preds = extend_preds(interaction_graph.graph, pivot, preds)
        results.author_predictions[conv.id][STEM_PRPG_MODEL] = preds

        stance_interactions_graph = connect_stance_nodes(core_interactions, author_posts_preds, STANCE_EDGE_WEIGHT)
        pbar.set_description(f"Conversation {conv.id}, participants: {conv.number_of_participants}")
        maxcut_with_stance = get_maxcut_with_stance_results(stance_interactions_graph, conv.root.node_id, "weight")
        results.maxcut_with_stance_results[conv.id] = maxcut_with_stance
        # maxcut_with_stance.draw(outpath=f"graphs/{MCSN_CORE_MODEL}-{conv.id}.png")

        negative, positive = decide_stance_groups_by_stance_nodes(maxcut_with_stance.get_supporters(),
                                                                  maxcut_with_stance.get_complement())
        preds = get_author_preds(positive, negative)
        results.author_predictions[conv.id][MCSN_CORE_MODEL] = preds

        # propagate results from core to full graph
        preds = extend_preds(interaction_graph.graph, pivot, preds)
        results.author_predictions[conv.id][MCSN_PRPG_MODEL] = preds

    results.total_time.end()
    return results


def results_to_df(results: ExperimentResults) -> pd.DataFrame:
    records = []
    for conv_id, predictions in tqdm(results.author_predictions.items()):
        conv = results.convs_by_id[conv_id]
        n_authors = conv.number_of_participants
        topic = conv.root.data["topic_name"]
        author_labels = get_author_labels(conv)
        all_posts_labels = get_posts_labels(conv)
        confident_conv = conv_id in results.confident_convs
        confident_core = conv_id in results.confident_core_convs

        for model in MODELS:
            author_preds = predictions.get(model, None)
            if author_preds is None:
                continue

            posts_preds, posts_preds = get_posts_preds(conv, all_posts_labels, author_preds)
            for _, node in conv.iter_conversation():
                author_label = author_labels.get(node.author)
                post_label = all_posts_labels.get(node.node_id)
                if (author_label or post_label) is None:
                    continue

                record = {"node_id": node.node_id, "author": node.author, "conv_id": conv_id, "topic": topic,
                          "authors": n_authors, "posts": conv.size, "model": model,
                          "author_label": author_label, "author_pred": author_preds.get(node.author),
                          "post_label": post_label, "post_pred": posts_preds.get(node.node_id),
                          "confident_conv": confident_conv, "confident_core": confident_core}

                records.append(record)

    return pd.DataFrame.from_records(records)


def get_metrics(y_true: Sequence[int], y_pred: Sequence[int], suffix: str = None) -> dict:
    if suffix is None:
        suffix = ""
    else:
        suffix = f"-{suffix}"

    return {
        f"support": len(y_true),
        f"accuracy{suffix}": accuracy_score(y_true, y_pred),
        f"recall-1{suffix}": recall_score(y_true, y_pred, pos_label=1),
        f"precision-1{suffix}": precision_score(y_true, y_pred, pos_label=1),
        f"recall-0{suffix}": recall_score(y_true, y_pred, pos_label=0),
        f"precision-0{suffix}": precision_score(y_true, y_pred, pos_label=0)
    }


def eval_results_per_conversation(results_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    results_df = results_df.dropna()
    for (conv_id, model), conv_df in results_df.groupby(["conv_id", "model"]):
        topic = conv_df.iloc[0]["topic"]
        confident_conv = conv_df.iloc[0]["confident_conv"]
        confident_core = conv_df.iloc[0]["confident_core"]
        record = {"conv_id": conv_id, "topic": topic, "model": model, "confident": confident_conv,
                  "confident_core": confident_core}

        y_true, y_pred = conv_df["post_label"], conv_df["post_pred"]
        record.update(get_metrics(y_true, y_pred, suffix="post"))

        conv_df = conv_df.drop_duplicates("author")
        y_true, y_pred = conv_df["author_label"], conv_df["author_pred"]
        record.update(get_metrics(y_true, y_pred, suffix="author"))

        records.append(record)

    return pd.DataFrame.from_records(records)


def get_metric_columns(eval_df: pd.DataFrame) -> List[str]:
    return [col for col in eval_df.columns if (col.startswith("acc") or col.startswith("recall") or col.startswith("prec"))]


def calculate_metrics_record(topic_df: pd.DataFrame, metric_columns: List[str], topic: str, model: str,
                                   confident: Optional[str], core_confident: Optional[str]) -> dict:
    record = {"topic": topic, "model": model, "convs": len(topic_df), "confident": confident,
              "core_confident": core_confident}
    for metric in metric_columns:
        record.update({
            f"{metric}-macro": topic_df[metric].mean(),
            f"{metric}-std": topic_df[metric].std(),
            f"{metric}-weighted": np.average(topic_df[metric], weights=topic_df["support"])
        })
    return record


def eval_results_per_topic(conv_eval_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = get_metric_columns(conv_eval_df)
    records = []
    for (topic, model, confident, core_confident), topic_df in conv_eval_df.groupby(["topic", "model", "confident",
                                                                                     "confident_core"]):
        record = calculate_metrics_record(topic_df, metric_columns, topic, model, confident, core_confident)
        records.append(record)

    for (topic, model, confident), topic_df in conv_eval_df.groupby(["topic", "model", "confident"]):
        record = calculate_metrics_record(topic_df, metric_columns, topic, model, confident, None)
        records.append(record)

    for (topic, model, core_confident), topic_df in conv_eval_df.groupby(["topic", "model", "confident_core"]):
        record = calculate_metrics_record(topic_df, metric_columns, topic, model, None, core_confident)
        records.append(record)

    for (topic, model), topic_df in conv_eval_df.groupby(["topic", "model"]):
        record = calculate_metrics_record(topic_df, metric_columns, topic, model, None, None)
        records.append(record)

    for model, topic_df in conv_eval_df.groupby(["model"]):
        record = calculate_metrics_record(topic_df, metric_columns, None, model, None, None)
        records.append(record)

    for (model, confident), topic_df in conv_eval_df.groupby(["model", "confident"]):
        record = calculate_metrics_record(topic_df, metric_columns, None, model, confident, None)
        records.append(record)

    for (model, core_confident), topic_df in conv_eval_df.groupby(["model", "confident_core"]):
        record = calculate_metrics_record(topic_df, metric_columns, None, model, None, core_confident)
        records.append(record)

    for (model, confident, core_confident), topic_df in conv_eval_df.groupby(["model", "confident", "confident_core"]):
        record = calculate_metrics_record(topic_df, metric_columns, None, model, confident, core_confident)
        records.append(record)

    return pd.DataFrame.from_records(records).sort_values(by=["topic", "model", "confident", "core_confident"])


def is_relevant_conversation(conv: Conversation) -> bool:
    return bool(get_author_labels(conv))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--results_path", "-r", type=str, required=True,
                               help="path of the intermediate result file (pred per post)")
    parent_parser.add_argument("--outpath", "-o", type=str,
                               help="path of the eval output file (eval per topic)")

    process_parser = subparsers.add_parser("process", parents=[parent_parser])
    process_parser.add_argument("path", type=str,
                                help="Path to the IAC directory containing all dataset as downloaded and extracted")
    process_parser.add_argument("--naive", const=True, action="store_const", default=False,
                                help="Apply only greedy and zero-shot approaches. SDP-based models won't be processed.")

    eval_parser = subparsers.add_parser("eval", parents=[parent_parser])

    args = parser.parse_args()

    if args.action == "process":
        zero_shot_probs = get_zs_post_labels("../data/fourforums/4forums-raw-preds.compact.csv")
        zero_shot_labels = get_zs_post_labels("../data/fourforums/4forums-raw-preds.compact.csv", probs=False)
        print("loading conversations")
        convs = load_conversations_from_dataframe(args.path)
        convs = list(filter(is_relevant_conversation, convs))
        decide_stance_groups = partial(decide_stance_groups_by_zs, zs_labels=zero_shot_labels, strategy="average")
        results = process_stance(convs, decide_stance_groups, zero_shot_probs, naive_results=args.naive)
        results.show()

        print("convert results to detailed_df")
        results_df = results_to_df(results)
        results_df.to_csv(args.results_path, index=False)

    elif args.action == "eval":
        results_df = pd.read_csv(args.results_path)

    else:
        raise RuntimeError("should not get here - no such option")

    print("evaluate_results")
    conv_eval_df = eval_results_per_conversation(results_df)
    conv_eval_df.to_csv("4forums-eval-conv.csv", index=False, float_format="%.3f")

    eval_df = eval_results_per_topic(conv_eval_df)
    if args.outpath is not None:
        print(f"save eval results to {args.outpath}")
        eval_df.to_csv(args.outpath, index=False, float_format="%.3f")

    print(eval_df)

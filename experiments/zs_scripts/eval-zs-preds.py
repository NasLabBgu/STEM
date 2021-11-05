import pandas as pd

from sklearn.metrics import accuracy_score, recall_score, precision_score

import argparse


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["topic_str", "label", "pred"])
    df["topic"] = df["topic_str"]
    del df["topic_str"]
    return df[df["label"] >= 0]


def accuracy(topic_data: pd.DataFrame) -> float:
    relevant_labels = topic_data[(topic_data["pred"] < 2) & (topic_data["label"] < 2)]
    return accuracy_score(relevant_labels["label"], relevant_labels["pred"])


def recall(topic_data: pd.DataFrame, pos_label: int) -> float:
    relevant_labels = topic_data[(topic_data["pred"] < 2) & (topic_data["label"] < 2)]
    return recall_score(relevant_labels["label"], relevant_labels["pred"], pos_label=pos_label)


def precision(topic_data: pd.DataFrame, pos_label: int) -> float:
    relevant_labels = topic_data[(topic_data["pred"] < 2) & (topic_data["label"] < 2)]
    return precision_score(relevant_labels["label"], relevant_labels["pred"], pos_label=pos_label)


def calculate_metrics(topic_data: pd.DataFrame) -> pd.Series:
    topic_size = len(topic_data)
    none_size = topic_data.loc[topic_data["pred"] == 2, "pred"].count()
    stats = {
        "size": topic_size,
        "num_none": none_size,
        "support_size": topic_size - none_size,
        "pos_size": topic_data.loc[topic_data["label"] == 1, "label"].count(),
        "neg_size": topic_data.loc[topic_data["label"] == 0, "label"].count(),
        "acc": accuracy(topic_data),
        "recall-pos": recall(topic_data, pos_label=1),
        "precision-pos": precision(topic_data, pos_label=1),
        "recall-neg": recall(topic_data, pos_label=0),
        "precision-neg": precision(topic_data, pos_label=0)
    }
    return pd.Series(stats, index=stats.keys())


def evaluate(data: pd.DataFrame):
    metrics = data.groupby("topic").apply(calculate_metrics)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(metrics)
        print()
        print(metrics.describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to dataset to evaluate. must contains 'topic', 'label' and 'pred' columns")

    args = parser.parse_args()

    data = load_data(args.path)
    evaluate(data)

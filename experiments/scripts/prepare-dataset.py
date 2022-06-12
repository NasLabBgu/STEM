import os
from typing import Iterable, Union, Tuple, Dict, Any, List
import argparse
import pandas as pd
from tqdm.auto import tqdm
from itertools import groupby, starmap

from experiments.datahandlers.iac.fourforum_labels import AuthorLabel
from experiments.datahandlers.loaders import load_conversations
from experiments.datahandlers.iac.fourforum_labels import load_author_labels as load_4forums_author_labels


def up_to_root() -> str:
    file = os.path.abspath(__file__)
    current_dir = os.path.dirname(file)
    while os.path.split(current_dir)[-1] != "cmv-stance-classification":
        current_dir = os.path.dirname(current_dir)

    return current_dir


try:
    import conversant
except:
    import sys
    import os

    # ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ROOT_DIR = up_to_root()
    CONVERSANT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "conversant")
    # EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
    # sys.path.append(ROOT_DIR)
    sys.path.append(CONVERSANT_DIR)

from conversant.conversation import Conversation
from conversant.conversation.conversation_utils import conversation_to_dataframe


UNKNOWN_TOPIC_ID = -1
FOURFORUMS_DIR = "fourforums"
FOURFORUMS_AUTHOR_LABELS_FILENAME = "mturk_author_stance.txt"

LabelsByConversation = Dict[Any, Dict[Any, int]]
PostLabels = Dict[Tuple[Any, Any], int]


def conversations_to_dataframe(conversations: Iterable[Conversation]) -> pd.DataFrame:
    return pd.concat(map(conversation_to_dataframe, tqdm(conversations)))


def has_topic(conversation: Conversation) -> bool:
    return conversation.root.data["topic"] != UNKNOWN_TOPIC_ID


def get_parent_post_id(dataset_name: str, row: pd.Series) -> Union[str, int]:
    if row["is_absolute_root"]:
        return -1

    return f"{dataset_name}_{row['conversation_id']}_{int(row['parent_id'])}"


def transform_dataframe(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    absolute_ids = data.apply(lambda row: f"{dataset_name}_{row['conversation_id']}_{row['node_id']}", axis=1)
    data.insert(0, "post_id", absolute_ids)
    del data["node_id"]

    absolute_parent_ids = data.apply(lambda row: get_parent_post_id(dataset_name, row), axis=1)
    data.insert(2, "parent_post_id", absolute_parent_ids)
    del data["parent_id"]

    data["text"] = data["data.text"]
    data.loc[data["text"].str.len() == 0, "text"] = "[EMPTY]"
    data["topic_id"] = data["data.topic"]
    data["topic_name"] = data["data.topic_name"]
    data["quote_source_ids"] = data["data.quote_source_ids"]
    relevant_columns = list(filter(lambda col: not col.startswith("data."), data.columns))
    return data[relevant_columns]

# LABELS

def create_author_labels_dict(labels: Iterable[AuthorLabel]) -> Dict[Any, int]:
    return {l.author_id: l.stance - 2 for l in labels if l.stance is not None}


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


def add_labels(
        df: pd.DataFrame,
        post_labels: PostLabels,
        author_labels_per_conversation: LabelsByConversation
) -> pd.DataFrame:

    def get_post_label(row: pd.Series) -> int:
        conv_id = row["conversation_id"]
        node_id = row["node_id"]
        return post_labels.get((conv_id, node_id), -1)

    def get_author_label(row: pd.Series) -> int:
        conv_id = row["conversation_id"]
        if conv_id not in author_labels_per_conversation:
            return -1

        conversation_labels = author_labels_per_conversation[conv_id]
        author = row["author"]
        return conversation_labels.get(author, -1)

    df.loc[:, "post_label"] = df.apply(get_post_label, axis=1)
    df.loc[:, "author_label"] = df.apply(get_author_label, axis=1)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")
    parser.add_argument("out", type=str,
                        help="Output path to store the dataset in the new format (similar to VAST)")
    parser.add_argument("--only-with-topic", "-t", type=bool, default=True,
                        help="indicates if to filter conversations without a3 known topic")

    args = parser.parse_args()

    convs = load_conversations(args.dataset, args.path)
    if args.only_with_topic:
        convs = list(filter(lambda conv: conv.root.data["topic"] != -1, convs))

    df = conversations_to_dataframe(convs)
    author_labels_per_conversation, post_labels_per_conversation = get_4forums_labels(args.path)
    df = add_labels(df, post_labels_per_conversation, author_labels_per_conversation)
    df = transform_dataframe(df, args.dataset)
    df.to_csv(args.out, index=False)

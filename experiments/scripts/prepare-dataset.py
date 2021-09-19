from typing import Iterable, Union
import argparse
import pandas as pd
from tqdm.auto import tqdm


UNKNOWN_TOPIC_ID = -1


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
from experiments.datahandlers.loaders import load_conversations


def conversations_to_dataframe(conversations: Iterable[Conversation]) -> pd.DataFrame:
    return pd.concat(map(conversation_to_dataframe, tqdm(conversations)))


def has_topic(conversation: Conversation) -> bool:
    return conversation.root.data["topic"] != UNKNOWN_TOPIC_ID


def get_parent_post_id(dataset_name: str, row: pd.Series) -> Union[str, int]:
    if row["is_absolute_root"]:
        return -1

    return f"{dataset_name}_{row['full_conv_id']}_{int(row['parent_id'])}"


def transform_dataframe(data: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    absolute_ids = data.apply(lambda row: f"{dataset_name}_{row['full_conv_id']}_{row['node_id']}", axis=1)
    data.insert(0, "post_id", absolute_ids)
    del data["node_id"]

    absolute_parent_ids = data.apply(lambda row: get_parent_post_id(dataset_name, row), axis=1)
    data.insert(2, "parent_post_id", absolute_parent_ids)
    del data["parent_id"]

    data["text"] = data["data.text"]
    data["topic_id"] = data["data.topic"]
    data["topic_name"] = data["data.topic_name"]
    data["quote_source_ids"] = data["data.quote_source_ids"]
    relevant_columns = list(filter(lambda col: not col.startswith("data."), data.columns))
    return data[relevant_columns]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")
    parser.add_argument("out", type=str,
                        help="Output path to store the dataset in the new format (similar to VAST)")
    parser.add_argument("--only-with-topic", "-t", type=bool, default=True,
                        help="indicates if to filter conversations without a known topic")

    args = parser.parse_args()

    convs = load_conversations(args.dataset, args.path)
    if args.only_with_topic:
        convs = list(filter(lambda conv: conv.root.data["topic"] != -1, convs))

    df = conversations_to_dataframe(convs)
    df = transform_dataframe(df, args.dataset)
    df.to_csv(args.out, index=False)

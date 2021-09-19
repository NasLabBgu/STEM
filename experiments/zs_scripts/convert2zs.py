#%%
from typing import Dict, TextIO, Tuple, Iterable

import json
import argparse

import pandas as pd

from tqdm.auto import tqdm

from experiments.utils.text_utils import parse_text, get_tokens_with_stopwords


# base_dir = "/Users/ronpick/studies/stance/alternative/createdebate_released"
from conversant.conversation import Conversation
from conversant.conversation.conversation_utils import conversation_to_dataframe
from experiments.datahandlers.loaders import load_conversations


def conversations_to_dataframe(conversations: Iterable[Conversation]) -> pd.DataFrame:
    df = pd.concat(map(conversation_to_dataframe, conversations))
    return df


def convert_to_zs_format(conversations: Iterable[Conversation]) -> pd.DataFrame:
    records = []
    for conv in tqdm(conversations):
        conv_df = conversation_to_dataframe(conv)
        for _, record in conv_df.iterrows():
            text = record["data.text"]
            parsed_doc = parse_text(text)
            tokens, pos = get_tokens_with_stopwords(parsed_doc)
            tokens_str = json.dumps(tokens)
            pos_str = json.dumps(pos)
            text_s = " ".join(" ".join(sent) + "." for sent in tokens)
            topic_name = record["data.topic_name"]
            if topic_name is None:
                continue

            topics = [topic_name]  # get_relevant_nps_from_text(parsed_doc)
            for topic in topics:
                topic_str = topic #" ".join(topic)
                vast_record = {
                    "author": record["data.author_id"],
                    "post": text,
                    "ori_topic": topic_str,
                    "ori_id": None,
                    "new_topic": topic_str,
                    "label": -1,
                    "type_idx": 1,
                    "new_id": f"cd-{record['data.post_id']}",
                    "arc_id": None,
                    "text": tokens_str,
                    "pos_text": pos_str,
                    "text_s": text_s,
                    "topic": json.dumps(topic),
                    "topic_str": topic_str,
                    "seen?": 0,
                    "contains_topic?": (topic_str in text) if topic_str is not None else 0
                }
                records.append(vast_record)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="name of the dataset to prepare")
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")
    parser.add_argument("out", type=str,
                        help="Output path to store the dataset in the new format (similar to VAST)")

    args = parser.parse_args()

    convs = load_conversations(args.dataset, args.path)
    zs_df = convert_to_zs_format(convs)
    zs_df.to_csv(args.out)

# %%
from functools import partial
from itertools import chain
from multiprocessing import Pool
from typing import Dict, TextIO, Tuple, Iterable, Sequence, List, Callable

import json
import argparse

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from experiments.utils.text_utils import parse_text, get_tokens_with_stopwords

ZS_COLS_ORDER = ["author", "post", "ori_topic", "ori_id", "new_topic", "label", "type_idx", "new_id", "arc_id",
                 "text", "pos_text", "text_s", "topic", "topic_str", "seen?", "contains_topic?"]


topic_mapping = {
    "gay": "gay marriage",
    "abortion": "abortion legislation",
    "gun": "gun control",
    "climate": "climate change",
    "death": "death penalty",
    "existence": "god existence",
    "marijuana": "marijuana legalization"
}


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def chunk(seq: Sequence, size: int) -> Iterable[Sequence]:
    yield from (seq[pos: pos + size] for pos in range(0, len(seq), size))


def process_chunk(seq: Sequence, func: Callable) -> list:
    return list(map(func, seq))


def apply_parallel(func: Callable, seq: Sequence, chunksize: int = 100) -> pd.Series:
    processed: List = [None for _ in range(len(seq))]
    with Pool() as p:
        with tqdm(total=len(seq)) as pbar:
            chunks = chunk(seq, chunksize)
            i = 0
            process = partial(process_chunk, func=func)
            for processed_chunk in p.imap(process, chunks):
                pbar.update(len(processed_chunk))
                for element in processed_chunk:
                    processed[i] = element
                    i += 1

    return pd.Series(processed)


def convert_to_zs_format(data: pd.DataFrame) -> pd.DataFrame:
    text = data["text"]
    print("Parsing Texts")
    parsed_doc = apply_parallel(parse_text, text, 100)
    print("Extract tokens and post-tags")
    tokens_with_pos = apply_parallel(get_tokens_with_stopwords, parsed_doc, 100)
    tokens_with_pos = pd.DataFrame(tokens_with_pos.tolist(), index=data.index)
    tokens = tokens_with_pos[0]
    pos = tokens_with_pos[1]
    tokens_str = tokens.apply(json.dumps)
    pos_str = pos.apply(json.dumps)
    text_s = tokens.apply(lambda token_list: " ".join(" ".join(sent) + "." for sent in token_list))
    topic_name = data["topic_name"]
    new_topic = topic_mapping.get(topic_name, topic_name)
    topic_tokens = new_topic.str.split()
    topic_cleaned = new_topic  # currently doesn't need more processing

    renames = {
        "post_id": "ori_id",
        "topic_name": "ori_topic",
    }
    data = data.rename(columns=renames)

    data.loc[:, "post"] = text
    data.loc[:, "new_topic"] = new_topic
    data.loc[:, "label"] = data["post_label"] - 2
    data.loc[:, "type_idx"] = 1
    data.loc[:, "new_id"] = np.arange(len(data))
    data.loc[:, "arc_id"] = None
    data.loc[:, "text"] = tokens_str
    data.loc[:, "pos_text"] = pos_str
    data.loc[:, "text_s"] = text_s
    data.loc[:, "topic"] = topic_tokens.apply(json.dumps)
    data.loc[:, "topic_str"] = topic_cleaned
    data.loc[:, "seen?"] = 0
    data.loc[:, "contains_topic?"] = topic_tokens.apply(
        lambda token_list: int(all(token in text for token in token_list)))
    return data[ZS_COLS_ORDER]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str,
                        help="Path to the IAC directory containing all dataset as downloaded and extracted")
    parser.add_argument("out", type=str,
                        help="Output path to store the dataset in the new format (similar to VAST)")

    args = parser.parse_args()

    print("Loading data")
    df = load_data(args.path)
    print(f"Loaded dataset with shape {df.shape}")
    print("Processing and converting dataset")
    zs_df = convert_to_zs_format(df)
    print(f"Writing dataset to file: {args.out}")
    zs_df.to_csv(args.out, index=False)
    print("Done!")

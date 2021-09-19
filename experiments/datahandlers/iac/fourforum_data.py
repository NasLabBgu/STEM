import os
import csv
from operator import itemgetter

from itertools import groupby
from typing import NamedTuple, Iterable, List, Union, Dict, Tuple

import tqdm

import pandas as pd

from conversant.conversation import Conversation
from conversant.interactions import InteractionsGraph
from stance_classification.data.iac import FourForumInteractionsBuilder
from stance_classification.data.iac.fourforum_conversation_parser import FourForumConversationParser

POSTS_FILENAME = "post.txt"
NULL_VALUE = "\\N"
ROOT_PARENT_ID = NULL_VALUE

QUOTES_FILENAME = "quote.txt"
QUOTE_DISCUSSION_ID_INDEX = 0
QUOTE_POST_ID_INDEX = 1
QUOTE_SOURCE_DISCUSSION_ID_INDEX = 6
QUOTE_SOURCE_POST_ID_INDEX = 7

QUOTE_NODE_FIELD = "quote_source_ids"


class PostRecord(NamedTuple):
    topic: int
    topic_name: str
    discussion_id: int
    post_id: int
    author_id: int
    creation_date: str
    parent: int
    parent_missing: bool
    text: str
    quote_source_ids: List[int] # post ids of the quotes contained in this ppost

    @staticmethod
    def from_csv_record(
            record: List[str],
            quotes_mapping: Dict[Tuple[int, int], List[int]] = None,
            topic_mapping: Dict[int, int] = None,
            texts_mapping: Dict[int, str] = None,
            topic_str_mapping: Dict[int, str] = None
    ) -> 'PostRecord':
        discussion_id = int(record[0])
        topic_id = topic_mapping.get(discussion_id, -1)
        post_id = int(record[1])
        quotes = quotes_mapping.get((discussion_id, post_id)) if quotes_mapping is not None else None
        text = texts_mapping.get(int(record[6]), "[deleted]") if texts_mapping is not None else ""
        return PostRecord(
            topic_id,
            topic_str_mapping.get(topic_id),
            discussion_id,
            post_id,
            int(record[2]),
            str(record[3]),
            PostRecord.get_parent_id(record[4]),
            bool(int(record[5])),
            text,
            quotes or []
        )

    @staticmethod
    def get_parent_id(raw_parent_id: str) -> Union[int, None]:
        if raw_parent_id == ROOT_PARENT_ID:
            return None

        return int(raw_parent_id)


def load_texts_map(data_dir: str) -> Dict[int, str]:
    print("load texts mapping")
    text_path = os.path.join(data_dir, "text.txt")
    with open(text_path, 'r') as f:
        texts_map = {}
        text_id, post_text = None, ""
        for i, line in enumerate(f):
            if text_id is None:
                split_line = line.strip().split('\t', 1)
                text_id = int(split_line[0])
                line_text = split_line[1] if len(split_line) > 1 else ""
            else:
                line_text = line.strip()

            if not line_text.endswith('\\'):
                texts_map[text_id] = post_text + line_text
                text_id, post_text = None, ""
                continue

            post_text += line_text[:-1] + "\n"

        return texts_map


def load_topic_str_mapping(data_dir: str) -> Dict[int, str]:
    topics_path = os.path.join(data_dir, "topic.txt")
    with open(topics_path, 'r') as f:
        lines = f.read().strip().split("\n")
        split_lines = map(lambda l: tuple(map(str.strip, l.strip().split())), lines)
        pairs = map(lambda split: (int(split[0].strip()), split[1].strip()), split_lines)
        return dict(pairs)


def load_topics_mapping(data_dir: str) -> Dict[int, int]:
    discussion_topic_path = os.path.join(data_dir, "discussion_topic.txt")
    with open(discussion_topic_path, 'r') as f:
        lines = f.read().strip().split("\n")
        split_lines = map(lambda l: tuple(map(str.strip, l.strip().split())), lines)
        pairs = map(lambda split: (int(split[0].strip()), int(split[1].strip())), split_lines)
        return dict(pairs)


def load_quotes(path: str) -> Dict[Tuple[int, int], List[int]]:
    """
    Takes the quotes table and create a mapping between a post id to source post ids from which quotes were taken to the post corresponds to the key post id.
    :param path:
    :return:
    """
    print("load quotes mapping")
    quotes_mapping = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for (discussion_id, post_id), records in groupby(reader, key=itemgetter(QUOTE_DISCUSSION_ID_INDEX, QUOTE_POST_ID_INDEX)):
            relevant_records = filter(lambda r: r[QUOTE_SOURCE_DISCUSSION_ID_INDEX] == discussion_id, records)
            relevant_records = filter(lambda r: r[QUOTE_SOURCE_POST_ID_INDEX] != NULL_VALUE, relevant_records)
            quotes = list(map(int, map(itemgetter(QUOTE_SOURCE_POST_ID_INDEX), relevant_records)))
            if len(quotes) == 0:
                continue
            quotes_mapping[(int(discussion_id), int(post_id))] = quotes

    return quotes_mapping


def load_post_records(dirpath: str) -> Iterable[PostRecord]:
    quotes_path = os.path.join(dirpath, QUOTES_FILENAME)
    quotes_mapping = load_quotes(quotes_path)
    topic_mapping = load_topics_mapping(dirpath)
    topic_str_mapping = load_topic_str_mapping(dirpath)
    text_mapping = load_texts_map(dirpath)

    def to_post_record(record: List[str]):
        return PostRecord.from_csv_record(record, quotes_mapping=quotes_mapping, topic_mapping=topic_mapping,
                                          texts_mapping=text_mapping, topic_str_mapping=topic_str_mapping)

    posts_path = os.path.join(dirpath, POSTS_FILENAME)
    with open(posts_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        posts = map(to_post_record, reader)
        yield from posts


def build_conversations(post_records: Iterable[PostRecord]) -> Iterable[Conversation]:
    parser = FourForumConversationParser()
    for discussion_id, posts in groupby(post_records, key=lambda r: r.discussion_id):
        conversation = parser.parse((discussion_id, posts))
        yield conversation


def build_interaction_graphs(convs: Iterable[Conversation]) -> Iterable[InteractionsGraph]:
    interactions_parser = FourForumInteractionsBuilder()
    return map(interactions_parser.build, convs)


if __name__ == "__main__":
    data_path = "/home/dev/data/stance/IAC/alternative/fourforums"
    records = load_post_records(data_path)
    convs = tqdm.tqdm(build_conversations(records))
    convs = iter(convs)
    c: Conversation
    c = next(convs)
    for d, n in c.iter_conversation():
        print(n.data["text"])

    # c: Conversation = next(convs)
    # nodes_with_quotes = [n for _, n in c.iter_conversation() if n.data[QUOTE_NODE_FIELD]]
    # for n in nodes_with_quotes:
    #     print(n.data)

    # def calcweight(x: PairInteractionsData) -> float:
    #     return
    # graphs = build_interaction_graphs(convs)
    # g: InteractionsGraph = next(graphs)
    # g: InteractionsGraph = next(graphs)
    # g.set_interaction_weights(lambda x: x['replies'] + x["quotes"])
    # for intr in g.interactions:
    #     print(intr)

    # print(nodes_with_quotes)
    # print(next(convs))

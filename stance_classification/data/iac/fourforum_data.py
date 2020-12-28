import os
import csv
from operator import itemgetter

from itertools import groupby
from typing import NamedTuple, Iterable, List, Union, Dict, Tuple

from conversant.conversation import Conversation
from conversant.conversation.parse import NamedTupleConversationReader

POSTS_FILENAME = "post.txt"
NULL_VALUE = "\\N"
ROOT_PARENT_ID = NULL_VALUE

QUOTES_FILENAME = "quote.txt"
QUOTE_DISCUSSION_ID_INDEX = 0
QUOTE_POST_ID_INDEX = 1
QUOTE_SOURCE_DISCUSSION_ID_INDEX = 6
QUOTE_SOURCE_POST_ID_INDEX = 7


class PostRecord(NamedTuple):
    discussion_id: int
    post_id: int
    author_id: int
    creation_date: str
    parent: int
    parent_missing: bool
    text_id: int
    quote_source_ids: List[int] # post ids of the quotes contained in this ppost

    @staticmethod
    def from_csv_record(record: List[str], quotes_mapping: Dict[Tuple[int, int], List[int]] = None):
        discussion_id = int(record[0])
        post_id = int(record[1])
        quotes = quotes_mapping.get((discussion_id, post_id)) if quotes_mapping is not None else None
        return PostRecord(
            discussion_id,
            post_id,
            int(record[2]),
            str(record[3]),
            PostRecord.get_parent_id(record[4]),
            bool(int(record[5])),
            int(record[6]),
            quotes or []
        )

    @staticmethod
    def get_parent_id(raw_parent_id: str) -> Union[int, None]:
        if raw_parent_id == ROOT_PARENT_ID:
            return None

        return int(raw_parent_id)


PARSE_STRATEGY = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent"
}

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

    def to_post_record(record: List[str]):
        return PostRecord.from_csv_record(record, quotes_mapping=quotes_mapping)

    posts_path = os.path.join(dirpath, POSTS_FILENAME)
    with open(posts_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        posts = map(to_post_record, reader)
        yield from posts


def build_conversations(post_records: Iterable[PostRecord]) -> Iterable[Conversation]:
    parser = NamedTupleConversationReader(PARSE_STRATEGY)
    for discussion_id, posts in groupby(post_records, key=lambda r: r.discussion_id):
        conversation = parser.parse(posts)
        yield conversation


if __name__ == "__main__":
    data_path = "/home/dev/data/stance/IAC/alternative/fourforums"
    records = load_post_records(data_path)
    convs = build_conversations(records)
    c: Conversation = next(convs)
    nodes_with_quotes = [n for _, n in c.iter_conversation() if n.data["quote_source_ids"]]
    for n in nodes_with_quotes:
        print(n.data)
    # print(nodes_with_quotes)
    # print(next(convs))

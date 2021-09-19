import os
import csv
from operator import itemgetter

from itertools import groupby
from typing import NamedTuple, Iterable, List, Union, Dict, Tuple

import tqdm

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
    topic: str
    discussion_id: int
    post_id: int
    author_id: int
    creation_date: str
    parent: int
    agreement: int
    stance: int
    stance_group: int
    quote_source_ids: List[int] # post ids of the quotes contained in this ppost

    @staticmethod
    def from_changli_record(record: List[str], quotes_mapping: Dict[Tuple[int, int], List[int]] = None):
        topic = record[0]
        discussion_id = int(record[10])
        post_id = int(record[11])
        parent_id = PostRecord.get_parent_id(record[13])
        parent_agreement = int(record[5]) if (record[5] is not None and len(record[5]) > 0) else None
        quotes = quotes_mapping.get((discussion_id, post_id)) if quotes_mapping is not None else None
        return PostRecord(
            topic,
            discussion_id,
            post_id,
            int(record[12]),
            "-",
            parent_id,
            parent_agreement,
            int(record[7]),
            int(record[8]),
            quotes or []
        )

    @staticmethod
    def get_parent_id(raw_parent_id: str) -> Union[int, None]:
        if raw_parent_id == ROOT_PARENT_ID:
            return None

        return int(raw_parent_id)

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


def load_post_records(changli_records_path: str, fourforum_dirpath: str) -> Iterable[PostRecord]:
    quotes_path = os.path.join(fourforum_dirpath, QUOTES_FILENAME)
    quotes_mapping = load_quotes(quotes_path)

    def to_post_record(record: List[str]):
        return PostRecord.from_changli_record(record, quotes_mapping=quotes_mapping)

    with open(changli_records_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        print(header)
        posts = map(to_post_record, reader)
        yield from posts


def build_conversations(post_records: Iterable[PostRecord]) -> Iterable[Conversation]:
    parser = FourForumConversationParser()
    for discussion_id, posts in groupby(post_records, key=lambda r: r.discussion_id):
        try:
            conversation = parser.parse((discussion_id, posts))
            yield conversation
        except ValueError as e:
            print(discussion_id)
            print(e)


def build_interaction_graphs(convs: Iterable[Conversation]) -> Iterable[InteractionsGraph]:
    interactions_parser = FourForumInteractionsBuilder()
    return map(interactions_parser.build, convs)


if __name__ == "__main__":
    fourforum_dirpath = "/home/dev/data/stance/IAC/alternative/fourforums"
    changli_path = "/home/dev/data/stance/chang-li/data/4forum/records.csv"
    records = load_post_records(changli_path, fourforum_dirpath)
    convs = list(tqdm.tqdm(build_conversations(records)))
    print(convs[0].id)
    print(convs[2].id)
    print(convs[3].id)
    c: Conversation = convs[4]
    print(c)

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

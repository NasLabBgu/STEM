import csv
from typing import Dict, NamedTuple, Optional, Iterable, Any, Tuple, List

import os
from itertools import groupby
from operator import itemgetter

NULL_VALUE = "\\N"
ROOT_PARENT_ID = NULL_VALUE

QUOTES_FILENAME = "quote.txt"
QUOTE_DISCUSSION_ID_INDEX = 0
QUOTE_POST_ID_INDEX = 1
QUOTE_SOURCE_DISCUSSION_ID_INDEX = 6
QUOTE_SOURCE_POST_ID_INDEX = 7

QUOTE_NODE_FIELD = "quote_source_ids"


class DiscussionRecord(NamedTuple):
    discussion_id: int
    url: str
    title: str
    op: Optional[int]
    description_text_id: Optional[int]

    @staticmethod
    def from_iterable(it_record: Iterable[Any]) -> 'DiscussionRecord':
        n_fields = 5
        record = list(it_record)
        record += [None for _ in range(n_fields - len(record))]
        discussion_id, url, title, op, description = list(record)
        op = -1 if op == NULL_VALUE else int(op)
        return DiscussionRecord(int(discussion_id), url, title, op, description)


class DiscussionMetadata(NamedTuple):
    discussion_id: int
    record: DiscussionRecord
    topic_id: Optional[int]
    topic_str: Optional[str]
    stance_names: Optional[Dict[int, str]]
    local_stance_mapping: Optional[Dict[int, int]]


def load_discussion_mapping(data_dir: str) -> Dict[int, DiscussionRecord]:
    path = os.path.join(data_dir, "discussion.txt")
    with open(path, 'r') as f:
        lines = f.read().strip().split("\n")
        records = map(lambda l: DiscussionRecord.from_iterable(map(str.strip, l.strip().split("\t"))), lines)
        return {d.discussion_id: d for d in records}


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


def load_topics_str_mapping(data_dir: str) -> Dict[int, str]:
    topics_path = os.path.join(data_dir, "topic.txt")
    with open(topics_path, 'r') as f:
        lines = f.read().strip().split("\n")
        split_lines = map(lambda l: tuple(map(str.strip, l.strip().split())), lines)
        pairs = map(lambda split: (int(split[0].strip()), split[1].strip()), split_lines)
        return dict(pairs)


def load_discussions_topic_mapping(data_dir: str) -> Dict[int, int]:
    discussion_topic_path = os.path.join(data_dir, "discussion_topic.txt")
    with open(discussion_topic_path, 'r') as f:
        lines = f.read().strip().split("\n")
        split_lines = map(lambda l: tuple(map(str.strip, l.strip().split())), lines)
        pairs = map(lambda split: (int(split[0].strip()), int(split[1].strip())), split_lines)
        return dict(pairs)


def load_topics_stances(data_dir: str) -> Dict[int, Dict[int, str]]:
    path = os.path.join(data_dir, "topic_stance.txt")
    with open(path, 'r') as f:
        lines = f.read().strip().split("\n")
        records = map(lambda l: tuple(map(str.strip, l.strip().split("\t"))), lines)
        return {int(topic_id): {int(r[1]): r[2] for r in topic_stance_records}
               for topic_id, topic_stance_records in groupby(records, key=itemgetter(0))
               }
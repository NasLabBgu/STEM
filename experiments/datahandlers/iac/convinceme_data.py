import os
import csv
from collections import Counter
from operator import itemgetter

from itertools import groupby
from typing import NamedTuple, Iterable, List, Union, Dict, Any, TypeVar, Optional, Type

import tqdm

from conversant.conversation import Conversation
from conversant.interactions import InteractionsGraph
from experiments.datahandlers.iac.iac_conversation_parser import IACPostRecord, IACConversationParser, \
    build_iac_conversations
from experiments.datahandlers.iac.iac_data_records import load_discussions_topic_mapping, load_topics_str_mapping, \
    load_texts_map, DiscussionMetadata, load_discussion_mapping, load_topics_stances, QUOTES_FILENAME, load_quotes, \
    ROOT_PARENT_ID

POSTS_FILENAME = "post.txt"
QUOTE_NODE_FIELD = "quote_source_ids"
NON_RELEVANT_STANCE_CLASSES = {"unknown", "other", "undecided"}

T = TypeVar('T', bound=NamedTuple)
N = TypeVar('N', bound=NamedTuple)


class DiscussionStanceRecord(NamedTuple):
    discussion_id: int
    discussion_stance_id: int
    discussion_stance: str
    topic_id: Optional[int]
    topic_stance_id: Optional[int]

    @staticmethod
    def from_iterable(it_record: Iterable[Any]) -> 'DiscussionStanceRecord':
        discussion_id, discussion_stance_id, discussion_stance, topic_id_str, topic_stance_id_str = list(it_record)
        topic_id = int(topic_id_str) if topic_id_str != "\\N" else None
        topic_stance_id = int(topic_stance_id_str) if topic_stance_id_str != "\\N" else None
        return DiscussionStanceRecord(int(discussion_id), int(discussion_stance_id), discussion_stance, topic_id,
                                      topic_stance_id)


def load_discussion_stance(data_dir: str, discussion_topic_mapping: Dict[int, int]) -> Dict[int, Dict[int, int]]:
    path = os.path.join(data_dir, "discussion_stance.txt")
    with open(path, 'r') as f:
        lines = f.read().strip().split("\n")
        records = map(lambda l: DiscussionStanceRecord.from_iterable(map(str.strip, l.strip().split("\t"))), lines)

        discussion_stance_mapping = {}
        for discussion_id, discussion_records in groupby(records, key=lambda r: r.discussion_id):
            discussion_records = list(discussion_records)
            topic_id = discussion_topic_mapping.get(discussion_id)
            stance_mapping = {r.discussion_stance_id: r.topic_stance_id for r in discussion_records if
                              r.topic_id == topic_id}
            discussion_stance_mapping[discussion_id] = stance_mapping

        return discussion_stance_mapping


def load_discussions_metadata(data_dirpath: str) -> Dict[int, DiscussionMetadata]:
    discussions_topic_mapping = load_discussions_topic_mapping(data_dirpath)
    topics_str_mapping = load_topics_str_mapping(data_dirpath)
    discussions_mapping = load_discussion_mapping(data_dirpath)
    topics_stance_names = load_topics_stances(data_dirpath)
    discussion_stance_mapping = load_discussion_stance(data_dirpath, discussions_topic_mapping)

    discussions_metadata = {}
    for d in discussions_mapping.values():
        discussion_id = d.discussion_id
        topic_id = discussions_topic_mapping.get(discussion_id)
        topic_name = topics_str_mapping[topic_id] if topic_id is not None else None
        metadata = DiscussionMetadata(discussion_id, d, topic_id, topic_name, topics_stance_names.get(topic_id, None),
                                      discussion_stance_mapping.get(discussion_id))
        discussions_metadata[discussion_id] = metadata

    return discussions_metadata


def create_discussion_artificial_root(discussion: DiscussionMetadata) -> IACPostRecord:
    title = discussion.record.title
    return IACPostRecord(
        discussion.topic_id, discussion.topic_str, discussion.discussion_id, -discussion.discussion_id,
        discussion.record.op, "", None, False, title, title, [], -1, "neutral", "root", discussion.record.url, -1,
        "unknown")


class ConvinceMeDataLoader:
    def __init__(self, data_dirpath: str):
        self.__text_mapping = load_texts_map(data_dirpath)
        self.__discussions_metadata = load_discussions_metadata(data_dirpath)
        self.__discussions_artifical_roots = {d.discussion_id: create_discussion_artificial_root(d)
                                              for d in self.__discussions_metadata.values()}
        self.__data_dirpath = data_dirpath

    def load_post_records(self) -> Iterable[IACPostRecord]:
        posts_path = os.path.join(self.__data_dirpath, POSTS_FILENAME)
        with open(posts_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            posts = map(self.__create_post_record, reader)
            for discussion_id, discussion_posts in groupby(posts, key=lambda p: p.discussion_id):
                yield self.__discussions_artifical_roots[discussion_id]
                yield from self.__infer_authors_stances(discussion_posts)

    def __create_post_record(self, record: List[str]) -> IACPostRecord:
        discussion_id = int(record[0])
        metadata = self.__discussions_metadata[discussion_id]
        topic_id = metadata.topic_id
        post_id = int(record[1])
        author_id = int(record[2])
        creation_date = str(record[3])
        quotes = []
        text = self.__text_mapping.get(int(record[6]), "[deleted]") if self.__text_mapping is not None else ""
        stance_id = int(record[8]) if record[8] != "\\N" else -1
        if (topic_id is not None) and (stance_id != -1):
            stance_id = metadata.local_stance_mapping.get(stance_id, -9)

        stance_name = self.__get_stance_name(discussion_id, stance_id)
        response_type = "rebuttal" if record[9] == '1' else "other"
        parent_id = self.__get_parent_id(record[4], discussion_id)
        parent_missing = bool(int(record[5])) or bool(parent_id)
        return IACPostRecord(topic_id, metadata.topic_str, discussion_id, post_id, author_id, creation_date, parent_id,
                             parent_missing, metadata.record.title, text, quotes, stance_id, stance_name, response_type, metadata.record.url,
                             -1, "unknown")

    def __get_parent_id(self, raw_parent_id: str, discussion_id: int) -> int:
        if raw_parent_id == ROOT_PARENT_ID:
            return self.__discussions_artifical_roots[discussion_id].post_id

        return int(raw_parent_id)

    def __get_stance_name(self, discussion_id: int, stance_id: int) -> Optional[str]:
        stance_names = self.__discussions_metadata[discussion_id].stance_names
        if stance_names is None:
            return None

        return stance_names[stance_id] if stance_id >= 0 else "unknown" if stance_id == -1 else "other"

    def __infer_authors_stances(self, discussion_records: Iterable[IACPostRecord]) -> Iterable[IACPostRecord]:
        records = list(discussion_records)
        discussion_id = records[0].discussion_id
        authors_stances: Dict[Any, List[int]] = {}
        stance_names = {}
        for record in records:
            author = record.author_id
            posts_stances = authors_stances.setdefault(author, [])
            post_stance = record.stance_id
            if post_stance >= 0:
                posts_stances.append(post_stance)
                if post_stance not in stance_names:
                    stance_names[post_stance] = record.stance_name

        # Aggregate stance per author
        authors_agg_stance = {author: get_most_common(stances) if len(stances) > 0 else None
                              for author, stances in authors_stances.items()}

        for record in records:
            author_stance_id = authors_agg_stance[record.author_id]
            author_stance_name = self.__get_stance_name(discussion_id, author_stance_id) if author_stance_id is not None else "unknown"
            new_record = copy_namedtuple(record, author_stance_id=author_stance_id, author_stance_name=author_stance_name)
            yield new_record


# def build_interaction_graphs(convs: Iterable[Conversation]) -> Iterable[InteractionsGraph]:
#     interactions_parser = FourForumInteractionsBuilder()
#     return map(interactions_parser.build, convs)


class AuthorStance(NamedTuple):
    discussion_id: int
    author_id: int
    topic_id: Optional[int]
    stance_id: Optional[int]
    stance_name: Optional[str]


K = TypeVar('K')


def get_most_common(it: Iterable[K]) -> K:
    return Counter(filter(lambda e: e is not None, it)).most_common(1)[0][0]


def copy_namedtuple(nt: N, **modifications: Any) -> N:
    attrs = nt._asdict()
    for k, v in modifications.items():
        if k in attrs:
            attrs[k] = v

    return type(nt)(**attrs)


if __name__ == "__main__":
    data_path = "../../../experiments/data/convinceme"
    loader = ConvinceMeDataLoader(data_path)
    records = loader.load_post_records()
    convs = tqdm.tqdm(build_iac_conversations(records))
    convs = iter(convs)
    c: Conversation
    c = next(convs)
    labeled_conv = False

    # authors_stance = {}
    # while not labeled_conv:
    #     c = next(convs)
    #     print(c.size)
    #     authors_stance = infer_authors_stances(c)
    #     labeled_conv = any(node.node_data.data["stance_id"] >= 0 for _, node in c.iter_conversation())
        # labeled_conv = any(astance.stance_id is not None for astance in authors_stance.values())

    # print(authors_stance)
    for d, n in c.iter_conversation():
        print("-" * d, n.node_id, f"({n.author}-{n.node_data.data['author_stance_name']})", f"^{n.parent_id}")

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

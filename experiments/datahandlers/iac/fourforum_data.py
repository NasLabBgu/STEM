import os
import csv
from collections import Counter

from itertools import groupby
from typing import Iterable, List, Union, Dict, Any

import tqdm

from conversant.conversation import Conversation
from conversant.interactions import InteractionsGraph
from experiments.datahandlers.iac.createdebate_data import AuthorStance
from experiments.datahandlers.iac.fourforum_interactions import FourForumInteractionsBuilder
from experiments.datahandlers.iac.fourforum_labels import load_author_labels, AuthorLabel
from experiments.datahandlers.iac.iac_conversation_parser import IACPostRecord, build_iac_conversations
from experiments.datahandlers.iac.iac_data_records import DiscussionMetadata, load_discussions_topic_mapping, \
    load_topics_str_mapping, load_discussion_mapping, load_topics_stances, load_texts_map, load_quotes,\
    QUOTES_FILENAME, ROOT_PARENT_ID

POSTS_FILENAME = "post.txt"



def load_discussions_metadata(data_dirpath: str) -> Dict[int, DiscussionMetadata]:
    discussions_topic_mapping = load_discussions_topic_mapping(data_dirpath)
    topics_str_mapping = load_topics_str_mapping(data_dirpath)
    discussions_mapping = load_discussion_mapping(data_dirpath)
    topics_stance_names = load_topics_stances(data_dirpath)

    discussions_metadata = {}
    for d in discussions_mapping.values():
        discussion_id = d.discussion_id
        topic_id = discussions_topic_mapping.get(discussion_id)
        metadata = DiscussionMetadata(discussion_id, d, topic_id, topics_str_mapping.get(topic_id),
                                      topics_stance_names.get(topic_id, None), None)
        discussions_metadata[discussion_id] = metadata

    return discussions_metadata


def load_discussion_authors_stance(path: str) -> Dict[int, Dict[int, AuthorLabel]]:
    authors_stance = load_author_labels(path)
    return {discussion_id: {a.author_id: a for a in authors_label}
            for discussion_id, authors_label in groupby(authors_stance, key=lambda d: d.discussion_id)}


class FourForumsDataLoader:
    AUTHOR_LABELS_FILENAME = "mturk_author_stance.txt"

    def __init__(self, data_dirpath: str):
        quotes_path = os.path.join(data_dirpath, QUOTES_FILENAME)
        self.__quotes_mapping = load_quotes(quotes_path)
        self.__text_mapping = load_texts_map(data_dirpath)
        self.__author_labels = load_discussion_authors_stance(os.path.join(data_dirpath, self.AUTHOR_LABELS_FILENAME))
        self.__discussions_metadata = load_discussions_metadata(data_dirpath)

        self.__data_dirpath = data_dirpath

    def load_post_records(self) -> Iterable[IACPostRecord]:
        posts_path = os.path.join(self.__data_dirpath, POSTS_FILENAME)
        with open(posts_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            posts = map(self.__create_post_record, reader)
            for discussion_id, discussion_posts in groupby(posts, key=lambda p: p.discussion_id):
                yield from discussion_posts

    def __create_post_record(self, record: List[str]) -> IACPostRecord:
        discussion_id = int(record[0])
        metadata = self.__discussions_metadata[discussion_id]
        topic_id = metadata.topic_id
        post_id = int(record[1])
        author_id = int(record[2])
        creation_date = str(record[3])
        quotes = self.__quotes_mapping.get((discussion_id, post_id)) if self.__quotes_mapping is not None else []
        text = self.__text_mapping.get(int(record[6]), "[deleted]") if self.__text_mapping is not None else ""
        stance_id = self.__get_post_stance_id(discussion_id, author_id)
        stance_name = metadata.stance_names[stance_id] if stance_id >= 0 else "other"
        parent_id = self.__get_parent_id(record[4])
        parent_missing = bool(int(record[5])) or bool(parent_id)
        return IACPostRecord(topic_id, metadata.topic_str, discussion_id, post_id, author_id, creation_date, parent_id,
                             parent_missing, text, quotes, stance_id, stance_name, None, metadata.record.url)

    def __get_post_stance_id(self, discussion_id: int, author_id: int) -> int:
        discussion_stances = self.__author_labels.get(discussion_id)
        if discussion_stances is None:
            return -1

        author_stance = discussion_stances.get(author_id)
        if author_stance is None:
            return -1

        return author_stance.stance

    @staticmethod
    def __get_parent_id(raw_parent_id: str) -> Union[int, None]:
        if raw_parent_id == ROOT_PARENT_ID:
            return None

        return int(raw_parent_id)


def get_most_common(it: Iterable[Any]) -> Any:
    return Counter(filter(lambda e: e is not None, it)).most_common(1)[0][0]


def infer_authors_stances(conv: Conversation) -> Dict[Any, AuthorStance]:
    topic = conv.root.node_data.data["topic"]
    discussion_id = conv.id
    authors_stances: Dict[Any, List[int]] = {}
    stance_names = {}
    for _, node in conv.iter_conversation():
        author = node.author
        posts_stances = authors_stances.setdefault(author, [])
        post_stance = node.node_data.data["stance_id"]
        if post_stance >= 0:
            posts_stances.append(post_stance)
            if post_stance not in stance_names:
                stance_names[post_stance] = node.node_data.data["stance_name"]

    # Aggregate stance per author
    authors_agg_stance = {author: get_most_common(stances) if len(stances) > 0 else None
                          for author, stances in authors_stances.items()}

    return {author: AuthorStance(discussion_id, author, topic, stance, stance_names.get(stance))
            for author, stance in authors_agg_stance.items()}


def build_interaction_graphs(convs: Iterable[Conversation]) -> Iterable[InteractionsGraph]:
    interactions_parser = FourForumInteractionsBuilder()
    return map(interactions_parser.build, convs)


if __name__ == "__main__":
    data_path = "/Users/ronpick/workspace/cmv-stance-classification/experiments/data/fourforums"
    loader = FourForumsDataLoader(data_path)
    records = loader.load_post_records()
    convs = tqdm.tqdm(build_iac_conversations(records))
    convs = iter(convs)
    c: Conversation
    labeled_conv = False
    authors_stance = {}
    while not labeled_conv:
        c = next(convs)
        print(c.size)
        authors_stance = infer_authors_stances(c)
        labeled_conv = any(node.node_data.data["stance_id"] >= 0 for _, node in c.iter_conversation())
        # labeled_conv = any(astance.stance_id is not None for astance in authors_stance.values())

    print(authors_stance)

    # c = next(convs)
    # for d, n in c.iter_conversation():
    #     print(n.data["text"])

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

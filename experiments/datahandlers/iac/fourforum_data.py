from typing import Iterable, List, Dict, Sequence, Optional
import os
import csv
from itertools import groupby

import tqdm

from conversant.conversation import Conversation
from conversant.interactions import InteractionsGraph
from experiments.datahandlers.iac.fourforum_interactions import FourForumInteractionsBuilder
from experiments.datahandlers.iac.fourforum_labels import load_author_labels, AuthorLabel
from experiments.datahandlers.iac.iac_conversation_parser import build_iac_conversations, IACRecordsLoader
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


class FourForumsDataLoader(IACRecordsLoader):

    AUTHOR_LABELS_FILENAME = "mturk_author_stance.txt"

    def __init__(self, data_dirpath: str):
        super().__init__()
        self.__data_dirpath = data_dirpath
        self.__quotes_mapping = load_quotes(os.path.join(data_dirpath, QUOTES_FILENAME))
        self.__text_mapping = load_texts_map(data_dirpath)
        self.__author_labels = load_discussion_authors_stance(os.path.join(data_dirpath, self.AUTHOR_LABELS_FILENAME))
        self.__discussions_metadata = load_discussions_metadata(data_dirpath)

    def iter_raw_records(self) -> Iterable[Sequence[str]]:
        posts_path = os.path.join(self.__data_dirpath, POSTS_FILENAME)
        with open(posts_path, 'r') as f:
            yield from csv.reader(f, delimiter='\t')

    def get_discussion_metadata(self, discussion_id: int):
        return self.__discussions_metadata[discussion_id]

    def get_text_mapping(self) -> Dict[int, str]:
        return self.__text_mapping

    def get_quotes(self, discussion_id: int, post_id: int) -> List[int]:
        return self.__quotes_mapping.get((discussion_id, post_id)) if self.__quotes_mapping is not None else []

    def get_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        return self.get_author_stance_id(author_id, discussion_id, post_id, record)

    def get_stance_name(self, discussion_id: int, stance_id: int) -> str:
        metadata = self.get_discussion_metadata(discussion_id)
        return metadata.stance_names[stance_id] if stance_id >= 0 else "unknown" if stance_id == -1 else "other"

    def get_response_type(self, record: Sequence[str]) -> str:
        return "reply"

    def get_author_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        discussion_stances = self.__author_labels.get(discussion_id)
        if discussion_stances is None:
            return -1

        author_stance = discussion_stances.get(author_id)
        if author_stance is None:
            return -1

        return author_stance.stance


def build_interaction_graphs(convs: Iterable[Conversation]) -> Iterable[InteractionsGraph]:
    interactions_parser = FourForumInteractionsBuilder()
    return map(interactions_parser.build, convs)


if __name__ == "__main__":
    data_path = "/Users/ronpick/workspace/cmv-stance-classification/experiments/data/fourforums"
    loader = FourForumsDataLoader(data_path)
    records = loader.load_post_records()
    convs = tqdm.tqdm(build_iac_conversations(records))
    convs = iter(convs)
    c: Conversation = next(convs)

    for d, n in c.iter_conversation():
        print("-" * d, n.node_id, f"({n.author}-{n.node_data.data['author_stance_name']})", f"^{n.parent_id}")

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

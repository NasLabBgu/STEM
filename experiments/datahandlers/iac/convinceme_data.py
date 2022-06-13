import os
import csv

from itertools import groupby
from typing import Iterable, List, Dict, Sequence, Optional

import tqdm

from conversant.conversation import Conversation
from experiments.datahandlers.iac.iac_conversation_parser import build_iac_conversations, \
    IACRecordsLoaderWithAuthorStanceInfer, IACRecordsLoader, RootlessIACRecordsLoader
from experiments.datahandlers.iac.iac_data_records import load_discussions_topic_mapping, load_topics_str_mapping, \
    load_texts_map, DiscussionMetadata, load_discussion_mapping, load_topics_stances, QUOTES_FILENAME, load_quotes, \
    DiscussionStanceRecord

POSTS_FILENAME = "post.txt"
QUOTE_NODE_FIELD = "quote_source_ids"
NON_RELEVANT_STANCE_CLASSES = {"unknown", "other", "undecided"}


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


class ConvinceMePartialDataLoader(RootlessIACRecordsLoader):

    STANCE_ID_INDEX = 8

    def __init__(self, data_dirpath: str):
        super().__init__()
        self.__data_dirpath = data_dirpath
        self.__text_mapping = load_texts_map(data_dirpath)
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
        return []

    def get_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        metadata = self.get_discussion_metadata(discussion_id)
        raw_stance_id = record[self.STANCE_ID_INDEX]
        stance_id = int(raw_stance_id) if raw_stance_id != self.NULL_VALUE else -1
        if (metadata.topic_id is not None) and (stance_id != -1):
            stance_id = metadata.local_stance_mapping.get(stance_id, self.OTHER_STANCE_VALUE)

        return stance_id

    def get_stance_name(self, discussion_id: int, stance_id: int) -> Optional[str]:
        stance_names = self.__discussions_metadata[discussion_id].stance_names
        if stance_names is None:
            return None

        if stance_id >= 0:
            return stance_names[stance_id]
        if stance_id == self.UNKNOWN_STANCE_VALUE:
            return self.UNKNOWN_STANCE_NAME

        return self.OTHER_STANCE_NAME

    def get_response_type(self, record: Sequence[str]) -> str:
        return "rebuttal" if record[9] == '1' else "other"

    def get_author_stance_id(self, author_id: int, discussion_id: int, post_id: int, record: Sequence[str]) -> int:
        return -1


def get_convinceme_records_loader(data_dirpath: str) -> IACRecordsLoader:
    partial_loader = ConvinceMePartialDataLoader(data_dirpath)
    return IACRecordsLoaderWithAuthorStanceInfer(partial_loader)


if __name__ == "__main__":
    data_path = "../../../experiments/data/convinceme"
    loader = get_convinceme_records_loader(data_path)
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

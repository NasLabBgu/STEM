from itertools import groupby
from typing import Tuple, NamedTuple, Iterable, Any, List, Optional

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser, NamedTupleConversationReader
from conversant.conversation.parse.conversation_parser import K, T
from conversant.conversation import Conversation

NamedTuples = Iterable[NamedTuple]

PARSE_STRATEGY = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent",
}

NO_PARENT_VALUE = None


class IACConversationParser(ConversationParser[Tuple[Any, NamedTuples], NamedTuple]):
    def __init__(self):
        self.namedtuple_parser = NamedTupleConversationReader(PARSE_STRATEGY, NO_PARENT_VALUE)

    def extract_conversation_id(self, raw_conversation: Tuple[Any, NamedTuples]) -> Any:
        return raw_conversation[0]

    def extract_node_data(self, raw_node: NamedTuple) -> NodeData:
        return self.namedtuple_parser.extract_node_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: Tuple[Any, NamedTuples]) -> Iterable[T]:
        return self.namedtuple_parser.iter_raw_nodes(raw_conversation[1])


class IACPostRecord(NamedTuple):
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
    stance_id: int
    stance_name: str
    response_type: Optional[str]
    url: str


def build_iac_conversations(post_records: Iterable[IACPostRecord]) -> Iterable[Conversation]:
    parser = IACConversationParser()
    for discussion_id, posts in groupby(post_records, key=lambda r: r.discussion_id):
        conversation = parser.parse((discussion_id, posts))
        yield conversation
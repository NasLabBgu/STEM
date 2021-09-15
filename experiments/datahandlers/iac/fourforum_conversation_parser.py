
from typing import Tuple, NamedTuple, Iterable, Any

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser, NamedTupleConversationReader
from conversant.conversation.parse.conversation_parser import K, T

NamedTuples = Iterable[NamedTuple]

PARSE_STRATEGY = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent",
}

NO_PARENT_VALUE = None


class FourForumConversationParser(ConversationParser[Tuple[Any, NamedTuples], NamedTuple]):
    def __init__(self):
        self.namedtuple_parser = NamedTupleConversationReader(PARSE_STRATEGY, NO_PARENT_VALUE)

    def extract_conversation_id(self, raw_conversation: Tuple[Any, NamedTuples]) -> Any:
        return raw_conversation[0]

    def extract_node_data(self, raw_node: NamedTuple) -> NodeData:
        return self.namedtuple_parser.extract_node_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: Tuple[Any, NamedTuples]) -> Iterable[T]:
        return self.namedtuple_parser.iter_raw_nodes(raw_conversation[1])

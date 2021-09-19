
from typing import Tuple, NamedTuple, Iterable, Any

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser, DataFrameConversationReader
from conversant.conversation.parse.conversation_parser import K, T

import pandas as pd

NamedTuples = Iterable[NamedTuple]

PARSE_STRATEGY = {
    "node_id": "post_id",
    "author": "author_id",
    "timestamp": "creation_date",
    "parent_id": "parent_post_id"
}

NO_PARENT_VALUE = None


class ConvinceMeConversationParser(ConversationParser[Tuple[Any, pd.DataFrame], pd.Series]):
    def __init__(self):
        self.df_parser = DataFrameConversationReader(PARSE_STRATEGY, NO_PARENT_VALUE)

    def extract_conversation_id(self, raw_conversation: Tuple[Any, pd.DataFrame]) -> Any:
        return raw_conversation[0]

    def extract_node_data(self, raw_node: pd.Series) -> NodeData:
        return self.df_parser.extract_node_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: Tuple[Any, pd.DataFrame]) -> Iterable[T]:
        return self.df_parser.iter_raw_nodes(raw_conversation[1])

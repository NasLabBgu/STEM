from typing import Iterable

import pandas as pd

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser


class TwitterConversationReader(ConversationParser[pd.DataFrame, pd.Series]):

    def __init__(self):
        super().__init__()

    def extract_node_data(self, raw_node: pd.Series, verbose=0) -> NodeData:
        node_id = raw_node.id_str
        if verbose > 0:
            print(f'parsing node id {node_id}')
        author = raw_node.user_id_str
        timestamp = raw_node.created_at
        data = dict(raw_node)
        parent_id = data.get('in_reply_to_status_id_str')
        if parent_id == 'nan':
            parent_id = None
            print(f'root node is {node_id}')

        return NodeData(node_id, author, timestamp, data, parent_id)

    def iter_raw_nodes(self, raw_conversation: pd.DataFrame) -> Iterable[pd.Series]:
        for x in raw_conversation.iterrows():
            yield x[1]
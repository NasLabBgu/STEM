from itertools import groupby
from operator import itemgetter
from typing import Union, List, Sequence, Iterable

import pandas as pd

from conversant.conversation.parse import ConversationParser, DataFrameConversationReader
from conversant.conversation import NodeData, ConversationNode, Conversation
from conversation.conversation_utils import iter_conversation_by_timestamp


class TwitterConversationReader(ConversationParser[pd.DataFrame, pd.Series]):
    """
    A class to parse a twitter conversation from a DataFrame as it was created from the twitter api.
    """

    DEFAULT_DATA = ['full_text', 'conversation_id_str', 'in_reply_to_screen_name']

    NODE_FIELDS_MAPPING = {
        "node_id": "id_str",
        "author": "user_id_str",
        "timestamp": "timestamp",
        "parent_id": "in_reply_to_status_id_str",
        "data": DEFAULT_DATA
    }

    ALL_DATA_FIELDS = "all"
    NO_DATA_FIELDS = "none"

    def __init__(self,
                 extra_data_fields: Union[None, str, List[str]] = None,
                 merge_sequential: bool = True
                 ):
        super(TwitterConversationReader, self).__init__()
        fields_mapping = {**self.NODE_FIELDS_MAPPING}
        if extra_data_fields == self.ALL_DATA_FIELDS:
            del fields_mapping["data"]
        elif extra_data_fields == self.NO_DATA_FIELDS:
            fields_mapping["data"] = []
        elif isinstance(extra_data_fields, Sequence):
            fields_mapping["data"] = extra_data_fields
        self.dataframe_parser = DataFrameConversationReader(fields_mapping, no_parent_value='nan')
        self.merge_sequential = merge_sequential

    def parse(self, raw_conversation: pd.DataFrame, root_id: str = None) -> Conversation:
        conversation = super(TwitterConversationReader, self).parse(raw_conversation, root_id)
        if self.merge_sequential:
            return merge_sequential_tweets(conversation)

        return conversation

    def extract_node_data(self, raw_node: pd.Series) -> NodeData:
        return self.dataframe_parser.extract_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: pd.DataFrame) -> Iterable[pd.Series]:
        raw_conversation["timestamp"] = raw_conversation["created_at"].apply(pd.Timestamp.timestamp)
        raw_conversation = merge_sequential_tweets(raw_conversation)
        for x in raw_conversation.iterrows():
            yield x[1]


def merge_sequential_tweets(conversation: Conversation) -> Conversation:
    ordered_tweets = map(itemgetter(1), iter_conversation_by_timestamp(conversation.root))
    for author, tweets in groupby(ordered_tweets, key=lambda n: n.author):
        merge_tweets(tweets)

    return conversation


def merge_tweets(tweets: Iterable[ConversationNode]) -> ConversationNode:
    tweets = iter(tweets)
    first_tweet = next(tweets)
    main_data = first_tweet.data
    for field in list(main_data.keys()):
        main_data[field] = [main_data[field]]

    main_children = {n.node_id: n for n in first_tweet.get_children()}
    for i, tweet in enumerate(tweets):
        if tweet.node_id in main_children:
            del main_children[tweet.node_id]

        current_data = tweet.data
        for field in current_data:
            if field not in main_data:
                main_data[field] = [None for _ in range(i)]

            main_data[field].append(current_data[field])

        current_children = {n.node_id: n for n in tweet.get_children()}
        main_children.update(current_children)

        tweet.chldren = []
        tweet.parent = None
        del tweet

    first_tweet.children = list(main_children.values())
    return first_tweet

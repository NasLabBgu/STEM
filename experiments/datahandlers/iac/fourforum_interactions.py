
from typing import List, Iterable, Tuple, Any, Union

from conversant.conversation import Conversation, ConversationNode
from conversant.interactions import InteractionsParser, InteractionsGraph
from conversant.interactions.aggregators import CountInteractionsAggregator
from stance_classification.data.cmv_agreement_aggregator import NodesInteractionsAggregator

QUOTE_NODE_FIELD = "quote_source_ids"


class FourForumInteractionsBuilder(object):
    def __init__(self):
        reply_counter = CountInteractionsAggregator("replies", get_reply_interaction_users)
        quotes_counter = CountInteractionsAggregator("quotes", get_quoted_users)
        nodes_aggregator = NodesInteractionsAggregator("nodes")

        self.__interactions_parser = InteractionsParser(reply_counter, quotes_counter, nodes_aggregator, directed=False)

    def build(self, conversation: Conversation) -> InteractionsGraph:
        return self.__interactions_parser.parse(conversation)


def get_reply_interaction_users(node: ConversationNode, branch: List[ConversationNode], *args) -> Iterable[Tuple[Any, Any]]:
    """

    :param node:
    :param branch:
    :param args:
    :return:
    """
    #TODO try to remove *args
    if len(branch) < 2:
        return []

    parent_author = branch[-2].author
    return [(node.author, parent_author)]


def get_quoted_users(node: ConversationNode, branch: List[ConversationNode], tree: Conversation) -> Iterable[Tuple[Any, Any]]:
    quotes_sources_post_ids = node.data[QUOTE_NODE_FIELD]

    quotes_authors = []
    for quote_source_post_id in quotes_sources_post_ids:
        author = find_quote_author(quote_source_post_id, tree, reversed(branch))
        if author is not None:
            quotes_authors.append(author)

    author = node.author
    return ((author, source_author) for source_author in quotes_authors)


def find_quote_author(source_post_id: str,
                      tree: Conversation,
                      preferred_nodes: Iterable[ConversationNode] = None,
                      ) -> Union[str, None]:

    if preferred_nodes is not None:
        for node in preferred_nodes:
            if node.node_id == source_post_id:
                return node.author

    for depth, node in tree.iter_conversation():
        if node.node_id == source_post_id:
            return node.author

    return None

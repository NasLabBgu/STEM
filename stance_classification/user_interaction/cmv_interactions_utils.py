from typing import Union, Iterable

from conversant.conversation import Conversation, NodeData
from stance_classification.utils import strip_quote_symbols, is_source_of_quote, find_user_mentions, \
    strip_mention_prefix

DELTA_BOT_USER = "DeltaBot"
CONFIRMED_AWARD_PREFIX = "Confirmed:"
REJECTED_AWARD_PREFIX = "This delta has been rejected"

AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"


def find_quote_author(quote: str,
                      tree: Conversation,
                      preferred_nodes: Iterable[NodeData] = None,
                      quote_timestamp: int = None
                      ) -> Union[str, None]:
    # clean quote symbols
    quote_text = strip_quote_symbols(quote)

    if preferred_nodes is not None:
        for node in preferred_nodes:
            node_data = node.data
            if is_source_of_quote(quote_text, node_data[TEXT_FIELD]):
                return node_data[AUTHOR_FIELD]

    for depth, node in tree.iter_conversation():
        node_data = node.data
        if quote_timestamp and (node_data[TIMESTAMP_FIELD] >= quote_timestamp):
            continue

        if is_source_of_quote(quote_text, node_data[TEXT_FIELD]):
            return node_data[AUTHOR_FIELD]


def check_delta_award(author: str, text: str) -> int:
    """
    check if the given text and author imply on delta award
    :param text:
    :param author:
    :return: 2 for a confirmed delta award, 1 for rejected delta award , -1 and 0 for no delta award.
    """
    if author == DELTA_BOT_USER:
        if text.startswith(CONFIRMED_AWARD_PREFIX):
            return 2

        if text.startswith(REJECTED_AWARD_PREFIX):
            return 1

    return 0


def find_award_recipient(delta_bot_text: str, award_status: int = -1) -> Union[str, None]:
    """
    extract the name of the awarded user.
    :param delta_bot_text: the award confirmation or rejection text by DeltaBot
    :param award_status: (optional) 2 if confirmed, 1 if rejected (affects on the type of the text to check).
                         if None is given the status will be concluded
    :return: name of the awarded user with the following special cases:
                OP - if the recipient is the OP
                None if the recipient is invalid (e.g DeltaBot)
    """
    if award_status != 1 and award_status != 2:
        raise ValueError(f"award_status can be 1 or 2. {award_status} was given")

    mentions = find_user_mentions(delta_bot_text)

    if len(mentions) == 0:
        if award_status == 1:
            if "OP" in delta_bot_text:
                return "OP"
            elif "DeltaBot" in delta_bot_text or "yourself" in delta_bot_text:
                return None

        raise Exception(f"no mentions found in award text: {delta_bot_text}")

    award_recipient_mention = mentions[0]
    username = strip_mention_prefix(delta_bot_text[slice(*award_recipient_mention)])
    return username

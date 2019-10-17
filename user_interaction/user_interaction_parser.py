from typing import Union, Dict, List, Iterable

from dataclasses import dataclass

from treetools.TreeTools import walk_tree
from utils import find_user_mentions, strip_mention_prefix, find_quotes, strip_quote_symbols, is_source_of_quote

DELTA_BOT_USER = "DeltaBot"
CONFIRMED_AWARD_PREFIX = "Confirmed:"
REJECTED_AWARD_PREFIX = "This delta has been rejected"

# node fields names
AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"



@dataclass
class UsersInteraction:
    num_replies: int = 0
    num_mentions: int = 0
    num_quotes: int = 0
    num_confirmed_delta_awards: int = 0
    num_rejected_delta_awards: int = 0


def parse_users_interactions(tree: dict):
    interactions: Dict[str, Dict[str: UsersInteraction]] = {}
    op: str
    current_branch_nodes: List[dict] = []
    for depth, node in walk_tree(tree):

        # check if first user (i.e post author)
        if len(current_branch_nodes) == 0:
            current_branch_nodes.append(node)
            op = node[AUTHOR_FIELD]
            continue

        if depth < len(current_branch_nodes):
            del current_branch_nodes[depth:]

        current_author = node[AUTHOR_FIELD]
        text = node[TEXT_FIELD]
        timestamp = node[TIMESTAMP_FIELD]

        ## checking award section TODO move to separate function
        # check if deltabot awarded a delta
        delta_award_status = check_delta_award(current_author, text)

        # add delta awards interactions
        if delta_award_status > 0:
            delta_award_recipient = find_award_recipient(text, delta_award_status)
            if delta_award_recipient == "OP":
                delta_award_recipient = op

            if delta_award_recipient is not None:
                if not (delta_award_recipient == current_branch_nodes[-2][AUTHOR_FIELD]
                        or current_branch_nodes[-2][AUTHOR_FIELD] == "[deleted]"):
                    delta_award_recipient = current_branch_nodes[-2][AUTHOR_FIELD]

                relevant_author = current_branch_nodes[-1][AUTHOR_FIELD]  # previous author
                author_interactions = interactions.setdefault(relevant_author, {})
                pair_interaction = author_interactions.setdefault(delta_award_recipient, UsersInteraction())
                if delta_award_status == 1:
                    pair_interaction.num_rejected_delta_awards += 1
                else:
                    pair_interaction.num_confirmed_delta_awards += 1

        ### End delta award section

        elif current_author == "DeltaBot":
            if "[doesn't necessarily mean a reversal]" not in text:
                print(f"DeltaBot: {text}")
        else:
            # parse current node interactions
            mentions_positions = find_user_mentions(text)
            user_mentions = [strip_mention_prefix(text[slice(*mention_pos)]) for mention_pos in mentions_positions]

            quotes_positions = find_quotes(text)
            quotes_authors = [
                find_quote_author(
                    text[slice(*q)], tree, reversed(current_branch_nodes), timestamp)
                for q in quotes_positions]

            # add interactions to graph
            prev_author = current_branch_nodes[-1][AUTHOR_FIELD]
            author_interactions = interactions.setdefault(current_author, {})

            # add reply interaction
            pair_interaction = author_interactions.setdefault(prev_author, UsersInteraction())
            pair_interaction.num_replies += 1

            # add mention interactions
            for user_mention in user_mentions:
                pair_interaction = author_interactions.setdefault(user_mention, UsersInteraction())
                pair_interaction.num_mentions += 1

            # add quotes interactions
            for quote_author in quotes_authors:
                pair_interaction = author_interactions.setdefault(quote_author, UsersInteraction())
                pair_interaction.num_quotes += 1

        current_branch_nodes.append(node)

    return interactions


def find_quote_author(quote: str, tree: dict, preferred_nodes: Iterable[dict] = None, quote_timestamp: int = None) -> Union[str, None]:
    """
    search the source of the quote in the tree.
    :param quote: the text of the quote to search.
    :param tree: the tree where the source text of the quote exists.
    :param preferred_nodes: (optional) optimize the process by searching this list of nodes first.
                            if None is given, the preferred nodes search is ignored.
    :param quote_timestamp: (optional) use timestamp to check only nodes with earlier timestamp.
                            if None is given, the timestamp of the node is ignored.
    :return: author of the quote if found, None if not found
    """
    # clean quote symbols
    quote_text = strip_quote_symbols(quote)

    if preferred_nodes is not None:
        for node in preferred_nodes:
            if is_source_of_quote(quote_text, node[TEXT_FIELD]):
                return node[AUTHOR_FIELD]

    for depth, node in walk_tree(tree):
        if quote_timestamp and (node[TIMESTAMP_FIELD] >= quote_timestamp):
            continue

        if is_source_of_quote(quote_text, node[TEXT_FIELD]):
            return node[AUTHOR_FIELD]


def check_delta_award(author: str, text: str):
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


def find_award_recipient(delta_bot_text: str, award_status: int = None) -> Union[str, None]:
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




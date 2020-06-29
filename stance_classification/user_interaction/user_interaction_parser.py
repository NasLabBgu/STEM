from typing import Union, Dict, List, Iterable, Any, Set

from dataclasses import dataclass, field as dataclass_field

from stance_classification.utils import find_user_mentions, strip_mention_prefix, find_quotes, strip_quote_symbols, \
    is_source_of_quote


DELTA_BOT_USER = "DeltaBot"
CONFIRMED_AWARD_PREFIX = "Confirmed:"
REJECTED_AWARD_PREFIX = "This delta has been rejected"

# node fields names
NODE_FIELD = "node"
AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"
LABELS_FIELD = "labels"

D = []


@dataclass
class UsersInteraction:
    num_replies: int = 0
    num_mentions: int = 0
    num_quotes: int = 0
    num_confirmed_delta_awards: int = 0
    num_rejected_delta_awards: int = 0
    labels: list = dataclass_field(default_factory=list)

    def __getitem__(self, field: str):
        return self.__dict__[field]


def add_award_interaction(author: str, text: str, op: str, current_branch_nodes: List[dict],
                          author_interactions: Dict[str, UsersInteraction]) -> bool:
    """
    add interaction from the user that gave the award to the recipient of the award
    :param author: the author of 'text'
    :param text: the text referring to the delta award.
    :param op: the author of the post
    :param current_branch_nodes: the previous nodes in this branch starting from the root to the current node.
    :param author_interactions: the interactions in the graph until this reply
    :return: return True if interaction found, False otherwise.
    """
    delta_award_status = check_delta_award(author, text)
    if delta_award_status == 0:
        return False

    delta_award_recipient = find_award_recipient(text, delta_award_status)
    if delta_award_recipient == "OP":
        delta_award_recipient = op

    if delta_award_recipient is None:
        return False

    if delta_award_recipient != current_branch_nodes[-2][AUTHOR_FIELD]:
        if current_branch_nodes[-2][AUTHOR_FIELD] != "[deleted]":
            print(delta_award_recipient, current_branch_nodes[-2][AUTHOR_FIELD])
            delta_award_recipient = current_branch_nodes[-2][AUTHOR_FIELD]

    pair_interaction = author_interactions.setdefault(delta_award_recipient, UsersInteraction())
    if delta_award_status == 1:
        pair_interaction.num_rejected_delta_awards += 1
    else:
        pair_interaction.num_confirmed_delta_awards += 1

    return True


def add_reply_interactions(parent_author: str, author_interactions: Dict[str, UsersInteraction]) -> bool:
    """
    add reply interaction from the author of the reply to parent node author.
    :param parent_author: the recipient of the reply
    :param author_interactions: interactions from the reply author to other users
    :return: return True if interaction found, False otherwise.
    """
    if parent_author == DELTA_BOT_USER:
        return False

    pair_interaction = author_interactions.setdefault(parent_author, UsersInteraction())
    pair_interaction.num_replies += 1


def add_mentions_interactions(text: str, author_interactions: Dict[str, UsersInteraction]) -> int:
    """
    parse mentions in the text and add interactions from 'author' to the mentioned user.
    :param text: reply content
    :param author_interactions: interactions from the author of 'text' to other users
    :return: return number of added interactions
    """
    mentions_positions = find_user_mentions(text)
    user_mentions = [strip_mention_prefix(text[slice(*mention_pos)]) for mention_pos in mentions_positions]
    if len(user_mentions) > 0:
        for user_mention in user_mentions:
            pair_interaction = author_interactions.setdefault(user_mention, UsersInteraction())
            pair_interaction.num_mentions += 1

        return len(user_mentions)


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


def add_labels(current_node: dict, parent_author: str, author_interactions: Dict[str, UsersInteraction]) -> bool:
    """
    extract the labels from the node if exists
    :param current_node:
    :param parent_author:
    :param author_interactions: interactions from the author of 'text' to other users
    :return: the labels dictionary if labels exists, otherwise empty dictionary
    """
    labels = current_node.setdefault(LABELS_FIELD, {})
    if len(labels) == 0:
        return False

    pair_interaction = author_interactions.setdefault(parent_author, UsersInteraction())
    pair_interaction.labels.append(labels)
    return True


# interaction weight functions
def cmv_interaction_weight(u: Any, v: Any, data: dict) -> float:
    return data["num_replies"] + data["num_quotes"] + (
                data["num_confirmed_delta_awards"] + data["num_rejected_delta_awards"]) * 3


IRRELEVANT_USERS = {None, "DeltaBot", "[deleted]"}


def is_relevant_user(source_user: str, op: str, source_users: Set[str], target_users: Set[str]) -> bool:
    if source_user in IRRELEVANT_USERS:
        return False

    irrelevant_users = IRRELEVANT_USERS | {source_user}
    filtered_targets = target_users - irrelevant_users
    filtered_sources = source_users - irrelevant_users
    if len(filtered_targets) == 0:
        return False

    if len(filtered_sources) == 0:
        return False

    # check if the user replied only to the op and the op didn't replied back
    if len(filtered_targets) == 1:
        if (op in target_users) and (op not in source_users):
            return False

    return True


def filter_interactions(users_interactions: Dict[str, Dict[str, UsersInteraction]], op: str):

    # build reversed interactions
    reversed_interactions = {user: set() for user in users_interactions.keys()}
    for source_user, interactions in users_interactions.items():
        for target_user in interactions.keys():
            sources = reversed_interactions.setdefault(target_user, set())
            sources.add(source_user)

    filtered_interactions = []
    for out_user, interactions in users_interactions.items():
        target_users = interactions.keys()
        source_users = reversed_interactions[out_user]

        if is_relevant_user(out_user, op, source_users, target_users):

            filtered_user_interactions = \
                {in_user: d for in_user, d in interactions.items()
                       if in_user not in IRRELEVANT_USERS and in_user != out_user
                 }

            filtered_interactions.append((out_user, filtered_user_interactions))

    return dict(filtered_interactions)











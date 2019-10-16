
import argparse

import json
from itertools import starmap
from typing import List, Iterable, Union, Dict

import numpy as np
import pandas as pd

from treetools.TreeTools import walk_tree
from user_interaction import UsersInteraction, check_delta_award, find_award_recipient
from utils import iter_trees_from_jsonl, find_user_mentions, find_quotes, strip_quote_symbols, QUOTE_START_SYMBOL, \
    is_source_of_quote, strip_mention_prefix

AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"

tmp: str


def analyze_data(trees: Iterable[dict]):

    for i, tree in enumerate(trees):
        print(f"Tree: {i}")
        interactions = build_user_interaction_graph(tree)
        # print(json.dumps(interactions, indent=4, default=lambda cls: cls.__dict__))


def build_user_interaction_graph(tree: dict):
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







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="trees in json format to analyze")

    # args = parser.parse_args()
    args = parser.parse_args(["/home/ron/data/bgu/trees_2.0.txt"])

    trees = iter_trees_from_jsonl(args.data)
    analyze_data(trees)



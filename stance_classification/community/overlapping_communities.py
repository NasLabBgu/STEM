from typing import Iterable, Set

from conversation import Conversation


def detect_overlapping_communities(conversation: Conversation) -> Iterable[Set[str]]:
    """
    generate sets of nodes (only their ids) that were detected as communities. the same node may occur in multiple communities.
    :param conversation:
    :return:
    """
    # create interaction graph
    # prune the root and create communities.
    # prune the node with the highest betweenness in the interaction and yield a modularity score
    # choose the split with the best modularity

    # create a tree of branched conversations with titles.
    pass
from functools import partial
from typing import Callable

import tqdm

from interactions.interactions_graph import PairInteractionsData, InteractionsGraph
from stance_classification.classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from stance_classification.reddit_conversation_parser import CMVConversationReader
from stance_classification.user_interaction.cmv_stance_interactions_graph_builder import \
    CMVStanceBasedInteractionGraphBuilder
from stance_classification.utils import iter_trees_from_lines


def cmv_stance_interaction_weight(pair_interaction: PairInteractionsData) -> float:
    interactions = pair_interaction.interactions
    return interactions["replies"] + interactions["quotes"] + interactions["deltas"]

def is_valid_interactions(interactions_graph: InteractionsGraph):
    return True


if __name__ == "__main__":
    # load trees
    trees_path = "/data/work/data/reddit_cmv/trees_2.0.txt"
    total_trees = sum(1 for _ in iter_trees_from_lines(trees_path))
    trees = tqdm.tqdm(iter_trees_from_lines(trees_path), total=total_trees)

    # convert trees into conversations
    cmv_reader = CMVConversationReader()
    conversations = map(cmv_reader.parse, trees)

    # convert conversation into interaction graphs
    iteractions_graph_builder = CMVStanceBasedInteractionGraphBuilder()
    maxcut_clf = MaxcutStanceClassifier()
    # interactions_graphs = map(iteractions_graph_builder.build, conversations)
    for conversation in conversations:
        op = conversation.root.author
        interactions_graph = iteractions_graph_builder.build(conversation)
        if not is_valid_interactions(interactions_graph):
            continue

        # apply max-cut partition on interaction graphs
        interactions_graph.set_interaction_weights(cmv_stance_interaction_weight)
        maxcut_clf.set_input(interactions_graph.graph)
        maxcut_clf.classify_stance(op)

        # create semi-annotated user pairs
        supporters = maxcut_clf.get_supporters()
        opposers = maxcut_clf.get_complement()
        for pair_data in interactions_graph.interactions:
            record = [
                conversation.root.node_id,
                pair_data.user1,
                pair_data.user2,
                ...
                # TODO create another CMVStanceBasedInteractionGraphBuilder
                #  with aggregator of texts between pair of users
                #  with info regarding who is the responder
                #  (technically just store common nodes of each pair of users in the conversation)
            ]


    # write into csv



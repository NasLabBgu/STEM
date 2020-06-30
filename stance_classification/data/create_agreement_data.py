import csv
from functools import partial
from typing import Callable, Set

import tqdm

from conversation import Conversation
from interactions import InteractionsParser
from stance_classification.data.cmv_agreement_aggregator import NodesInteractionsAggregator

from stance_classification.reddit_conversation_parser import CMVConversationReader

# intractions graph
from interactions.aggregators import CountInteractionsAggregator
from interactions.interactions_graph import PairInteractionsData, InteractionsGraph
from stance_classification.classifiers.maxcut_stance_classifier import MaxcutStanceClassifier
from stance_classification.user_interaction.cmv_stance_interactions_graph_builder import \
    CMVStanceBasedInteractionGraphBuilder, get_reply_interaction_users, get_mentioned_users, get_delta_users, \
    get_quoted_users
from stance_classification.utils import iter_trees_from_lines

IRRELEVANT_USERS = {None, "DeltaBot", "[deleted]", "AutoModerator"}

SUPPORT_GROUP_MARK = "S"
OPPOSE_GROUP_MARK = "O"
MAX_USERS_THRESHOLD = 80 # for efficiency reasons


class CMVAgreementBasedInteractionGraphBuilder(object):
    def __init__(self):
        reply_counter = CountInteractionsAggregator("replies", get_reply_interaction_users)
        mention_counter = CountInteractionsAggregator("mentions", get_mentioned_users)
        quotes_counter = CountInteractionsAggregator("quotes", get_quoted_users)
        delta_counter = CountInteractionsAggregator("deltas", get_delta_users)
        nodes_aggregator = NodesInteractionsAggregator("nodes")

        self.__interactions_parser = InteractionsParser(reply_counter, mention_counter, quotes_counter, delta_counter,
                                                        nodes_aggregator, directed=False)

    def build(self, conversation: Conversation) -> InteractionsGraph:
        return self.__interactions_parser.parse(conversation)


def cmv_stance_interaction_weight(pair_interaction: PairInteractionsData) -> float:
    interactions = pair_interaction.interactions
    return interactions["replies"] + interactions["quotes"] + interactions["deltas"]


def is_valid_interactions(interactions_graph: InteractionsGraph):
    if interactions_graph.graph.number_of_nodes() < 10:
        return False
    if interactions_graph.graph.number_of_nodes() > MAX_USERS_THRESHOLD:
        return False

    return True


def preprocess_interactions(interactions_graph: InteractionsGraph, op: str) -> InteractionsGraph:
    interactions_graph.filter_interactions(
        condition=lambda u: u.user1 not in IRRELEVANT_USERS and u.user2 not in IRRELEVANT_USERS,
        inplace=True
    )
    if op not in interactions_graph.graph.nodes:
        return interactions_graph

    interactions_graph.get_author_connected_component(op, inplace=True)
    interactions_graph.get_core_interactions(inplace=True)
    return interactions_graph

def generate_records(interactions_graph: InteractionsGraph, supporters: Set[str], opposers: Set[str], op: str, conversation: Conversation):
    # ["conv_id", "op", "author", "recipient", "author_group", "recipient_group", "n_replies",
    #  "n_quotes", "n_deltas", "n_mentions", "content", "parent_content", "agree"]
    conversation_id = conversation.root.node_id
    for pair_data in interactions_graph.interactions:
        for node in pair_data.interactions["nodes"]:
            author = node.author
            recipient = pair_data.user2 if author == pair_data.user1 else pair_data.user1
            author_group = "S" if author in supporters else "O"
            recipient_group = "S" if recipient in supporters else "O"
            label = int(author_group == recipient_group)
            parent_node_id = node.parent_id
            parent_content = None
            for depth, node in conversation.iter_conversation():
                if node.node_id == parent_node_id:
                    parent_content = node.data["text"]
            record = [
                conversation_id,
                op,
                author,
                recipient,
                author_group,
                recipient_group,
                pair_data.interactions["replies"],
                pair_data.interactions["quotes"],
                pair_data.interactions["deltas"],
                pair_data.interactions["mentions"],
                node.data["text"],
                parent_content,
                label
            ]
            yield record


if __name__ == "__main__":
    trees_path = "/data/work/data/reddit_cmv/trees_2.0.txt"
    outpath = "/data/work/data/reddit_cmv/cmv_users_agreement_v1.0.0.csv"

    f = open(outpath, 'w')
    writer = csv.writer(f, lineterminator='\n')
    output_header = ["conv_id", "op", "author", "recipient", "author_group", "recipient_group", "n_replies",
                  "n_quotes", "n_deltas", "n_mentions" ,"author_content", "parent_content", "agree"]
    writer.writerow(output_header)

    # load trees
    total_trees = sum(1 for _ in iter_trees_from_lines(trees_path))
    trees = tqdm.tqdm(iter_trees_from_lines(trees_path), total=total_trees)

    # convert trees into conversations
    cmv_reader = CMVConversationReader()
    conversations = map(cmv_reader.parse, trees)

    # convert conversation into interaction graphs
    iteractions_graph_builder = CMVAgreementBasedInteractionGraphBuilder()
    maxcut_clf = MaxcutStanceClassifier()
    # interactions_graphs = map(iteractions_graph_builder.build, conversations)
    valid_conversations = 0
    for conversation in conversations:
        op = conversation.root.author
        if op in IRRELEVANT_USERS:
            continue

        interactions_graph = iteractions_graph_builder.build(conversation)
        interactions_graph = preprocess_interactions(interactions_graph, op)
        if not is_valid_interactions(interactions_graph):
            continue

        valid_conversations += 1
        print(f"parse valid conversation number: {valid_conversations} - max cut on {interactions_graph.graph.order()} users")
        # apply max-cut partition on interaction graphs
        interactions_graph.set_interaction_weights(cmv_stance_interaction_weight)
        maxcut_clf.set_input(interactions_graph.graph)
        maxcut_clf.classify_stance(op)
        # maxcut_clf.draw()

        # create semi-annotated user pairs
        supporters = maxcut_clf.get_supporters()
        opposers = maxcut_clf.get_complement()
        records = generate_records(interactions_graph, supporters, opposers, op, conversation)

        # write into csv
        writer.writerows(records)
        f.flush()


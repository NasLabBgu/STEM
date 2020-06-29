import tqdm

from stance_classification.reddit_conversation_parser import CMVConversationReader
from stance_classification.user_interaction.cmv_stance_interactions_graph_builder import \
    CMVStanceBasedInteractionGraphBuilder
from stance_classification.utils import iter_trees_from_lines

if __name__ == "__main__":

    # load trees
    trees_path = "/data/work/data/reddit_cmv/trees_2.0.txt"
    total_trees = sum(1 for _ in iter_trees_from_lines(trees_path))
    trees = tqdm.tqdm(iter_trees_from_lines(trees_path), total=total_trees)

    # convert trees into conversations
    cmv_reader = CMVConversationReader()
    conversations = map(cmv_reader.parse, trees)

    # convert conversation into interaction graphs
    iteraction_graph_builder = CMVStanceBasedInteractionGraphBuilder()
    interaction_graphs = map(iteraction_graph_builder.build, conversations)

    # apply max-cut partition on interaction graphs


    # create semi-annotated user pairs


    # write into csv



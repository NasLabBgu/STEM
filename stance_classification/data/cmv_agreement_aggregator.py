from typing import List, Iterable, Tuple, Any

from conversation import NodeData, Conversation
from conversant.interactions.aggregators import InteractionsAggregator


class NodesInteractionsAggregator(InteractionsAggregator[NodeData, List[NodeData], List[NodeData]]):

    def __init__(self, interaction_name: str):
        super().__init__(interaction_name)

    def initialize_interactions_container(self) -> List[NodeData]:
        return list()

    def extract(self, node: NodeData, branch: List[NodeData], tree: Conversation) -> Iterable[Tuple[Any, Any, NodeData]]:
        if len(branch) <= 1:
            return []

        author = node.author
        recipient = branch[-2].author
        return [(author, recipient, node)]

    def add(self, u1: Any, u2: Any, interaction_value: NodeData, container: List[NodeData]):
        container.append(interaction_value)

    def aggregate(self, container: List[NodeData]) -> List[NodeData]:
        return container

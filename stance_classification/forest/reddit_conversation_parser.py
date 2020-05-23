from typing import Iterable

from anytree import AnyNode
from anytree.importer import JsonImporter

from conversant.conversation import NodeData
from conversant.conversation.parse import ConversationParser, AnyTreeConversationParser

JSON = str


class CMVConversationReader(ConversationParser[str, JSON]):
    """
    parse a cmv conversation written as json.

    Examples:
        the json data is as follows:
        >>> cmv_json = '''
        ... {
        ... 	"node": {
        ...        "text": "this is the content",
        ...        "id": "0",
        ...        "author": "OP",
        ...        "timestamp": 123
        ...      },
        ...      "children": [
        ...         {"node": {"id": "1", "author": "child", "timestamp": 124, "children": []}}
        ...     ]
        ...  }
        ... '''
        >>> cmv_parser = CMVConversationReader()
        >>> conversation = cmv_parser.parse(cmv_json)
        >>> print(conversation)
        OP - 0
        └── child - 1

    """

    def __init__(self):
        super(CMVConversationReader, self).__init__()
        self.__anytree_parser = AnyTreeConversationParser(extract_data)
        self.__importer = JsonImporter()

    def extract_node_data(self, raw_node: AnyNode) -> NodeData:
        return self.__anytree_parser.extract_data(raw_node)

    def iter_raw_nodes(self, raw_conversation: JSON) -> Iterable[AnyNode]:
        tree = self.__importer.import_(raw_conversation)
        return self.__anytree_parser.iter_raw_nodes(tree)


def extract_data(node: AnyNode) -> NodeData:
    data = node.node
    node_id = data["id"]
    author = data["author"]
    timestamp = data["timestamp"]

    parent_id = None
    if node.parent is not None:
        parent_id = node.parent.node["id"]

    return NodeData(node_id, author, timestamp, data, parent_id)

import csv
import json
import sys
from typing import Iterable, Tuple, NamedTuple, List

from stance_classification.utils import iter_trees_from_jsonl
from treetools.TreeTools import walk_tree


DELTA_BOT_USER = "DeltaBot"

NODE_FIELD = "node"
ID_FIELD = "id"
AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"

EXTRA_DATA_FIELD = "extra_data"
TITLE_FIELD = "title"


class AnnotationTask(NamedTuple):
    node_id: str
    tree_id: str
    title: str
    op: str
    op_text: str
    prev_replies: List[Tuple[str, str]]
    user_name: str
    current_reply: str
    reply_depth: int
    reply_timestamp: int

    @staticmethod
    def get_fields() -> tuple:
        return AnnotationTask._fields
        # return AnnotationTask(*[None for _ in range(10)])._fields


def iter_trees(trees_path: str) -> Iterable[dict]:
    yield from iter_trees_from_jsonl(trees_path)


def prepare_annotation_tasks_from_tree(tree: dict) -> Iterable[AnnotationTask]:
    first_node = tree[NODE_FIELD]
    tree_id = first_node["id"]
    tree_title = first_node[EXTRA_DATA_FIELD][TITLE_FIELD]
    op: str = first_node[AUTHOR_FIELD]
    op_text = first_node[TEXT_FIELD]
    current_branch_nodes: List[dict] = [first_node]  # Stores the previous nodes in the parsed branch
    current_branch_replies: List[Tuple[str, str]] = [(op, first_node[TEXT_FIELD])]  # Stores the previous nodes in the parsed branch

    tree_nodes = walk_tree(tree)
    next(tree_nodes)  # skip the first node
    for depth, node in tree_nodes:
        # check if the entire current branch was parsed, and start walking to the next branch
        if depth < len(current_branch_nodes):
            del current_branch_nodes[depth:]
            del current_branch_replies[depth:]

        text = node[TEXT_FIELD]
        timestamp = node[TIMESTAMP_FIELD]
        current_author = node[AUTHOR_FIELD]

        if current_author == DELTA_BOT_USER:
            continue
        else:
            task = AnnotationTask(node[ID_FIELD], tree_id, tree_title, op, op_text,
                                  current_branch_replies,
                                  current_author, text,
                                  depth, timestamp)

            yield task

        current_branch_nodes.append(node)
        current_branch_replies.append((node[AUTHOR_FIELD], node[TEXT_FIELD]))


def write_annotation_tasks(tasks: Iterable[AnnotationTask], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')

        # write header
        header = AnnotationTask.get_fields()
        writer.writerow(header)

        for task in tasks:
            record = list(task)
            record[5] = json.dumps(record[5])
            writer.writerow(record)


if __name__ == "__main__":

    labeled_trees_path = sys.argv[1]    # "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    outpath = sys.argv[2]    # "/home/ron/data/bgu/stance_annotation/tasks_v1.0.0.csv"

    trees = iter_trees(labeled_trees_path)
    tasks = (task for tree in trees for task in prepare_annotation_tasks_from_tree(tree))
    write_annotation_tasks(tasks, outpath)

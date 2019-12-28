import csv
import json
import sys
from collections import Counter
from itertools import islice
from random import shuffle
from typing import Iterable, Tuple, NamedTuple, List

from stance_classification.utils import iter_trees_from_jsonl
from treetools.TreeTools import walk_tree


DELTA_BOT_USER = "DeltaBot"
AUTO_MODERATOR_USER = "AutoModerator"

NODE_FIELD = "node"
ID_FIELD = "id"
AUTHOR_FIELD = "author"
TEXT_FIELD = "text"
TIMESTAMP_FIELD = "timestamp"

EXTRA_DATA_FIELD = "extra_data"
TITLE_FIELD = "title"

MAX_TEXT_LEN = 500
MAX_REPLIES_PER_USER = 5


class AnnotationTask(NamedTuple):
    node_id: str
    tree_id: str
    title: str
    op: str
    op_text: str
    prev_replies: str #List[Tuple[str, str]]
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
    current_branch_replies: List[Tuple[str, str]] = [(op, op_text)]  # Stores the previous nodes in the parsed branch

    users_replies_count = Counter()
    tree_nodes = walk_tree(tree, max_depth=6)
    next(tree_nodes)  # skip the first node
    for depth, node in tree_nodes:
        # check if the entire current branch was parsed, and start walking to the next branch
        if depth < len(current_branch_nodes):
            del current_branch_nodes[depth:]
            del current_branch_replies[depth:]

        text = node[TEXT_FIELD]
        timestamp = node[TIMESTAMP_FIELD]
        current_author = node[AUTHOR_FIELD]

        task = None
        if current_author == DELTA_BOT_USER or current_author == AUTO_MODERATOR_USER:
            pass
        elif current_author == op:
            pass
        elif users_replies_count[current_author] == MAX_REPLIES_PER_USER:
            pass
        else:
            branch_rerplies = format_branch_replies(current_branch_replies[1:][-3:])
            # current_reply = f"Author:: {current_author}\n{text}"
            task = AnnotationTask(node[ID_FIELD], tree_id, tree_title, op,
                                  format_prev_text(op_text),
                                  branch_rerplies,
                                  current_author, text,
                                  depth, timestamp)

        users_replies_count.update([current_author])
        current_branch_nodes.append(node)
        current_branch_replies.append((current_author, text))

        if task is not None:
            yield task


def format_prev_text(text: str):
    global MAX_TEXT_LEN

    if len(text) <= MAX_TEXT_LEN:
        return text

    splitted_text = text.split(" ")
    last_token_index = 0
    total_length = 0
    for i, token in enumerate(splitted_text):
        total_length += len(token) + 1
        if total_length >= MAX_TEXT_LEN:
            last_token_index = i
            break

    shorter_text = " ".join(splitted_text[:last_token_index]) + " ... "
    return shorter_text


def format_branch_replies(branch_replies: List[Tuple[str, str]]) -> str:
    branch_repr = "\n".join((f"Author:: {user}:\n{format_prev_text(reply)}\n" for user, reply in branch_replies))
    # last_user, last_reply = branch_replies[-1]
    # branch_repr += f"\n>>>{last_user}:\n{last_reply}"
    return branch_repr


def write_annotation_tasks(tasks: Iterable[AnnotationTask], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',', escapechar='"', quoting=csv.QUOTE_ALL, lineterminator='\r\n')

        # write header
        header = AnnotationTask.get_fields()
        writer.writerow(header)

        for task in tasks:
            record = list(task)
            record[5] = record[5]
            writer.writerow(record)


if __name__ == "__main__":

    labeled_trees_path = sys.argv[1]    # "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    outpath = sys.argv[2]    # "/home/ron/data/bgu/stance_annotation/tasks_v1.0.0.csv"

    trees = islice(iter_trees(labeled_trees_path), 1)
    tasks = list((task for tree in trees for task in prepare_annotation_tasks_from_tree(tree)))
    shuffle(tasks)
    write_annotation_tasks(tasks[:20], outpath)

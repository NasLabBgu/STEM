import csv
import json
import re
import sys
from collections import Counter, deque
from functools import partial
from itertools import islice, takewhile
from operator import itemgetter
from random import shuffle
from typing import Iterable, Tuple, NamedTuple, List, Set

from stance_classification.user_interaction.user_interaction_parser import parse_users_interactions
from stance_classification.user_interaction.users_interaction_graph import get_core_interactions
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

MAX_TEXT_LEN = 1000
MAX_DISPLAY_TEXT_LEN = 500
MAX_REPLIES_PER_USER = 5
MAX_DEPTH = 3
MAX_TITLE_LENGTH = 120
MAX_QUOTES = 1
MAX_QUOTE_PORTION = 0.5

MIN_CORE_SIZE = 7

QUOTE_TAGS_SIZE = len("<quote></quote>")

QUOTE_PATT = re.compile("<quote>(.+)</quote>", re.DOTALL)

BLOCKQUOTE_TEMPLATE = """<p style="font-size=16px; font-style: italic; margin-left: 32px; font-family: Georgia, 'Times New Roman', serif;">(a quote from a previous reply)<br><blockquote style="font-size=16px; font-style: italic; margin-left: 32px; font-family: Georgia, 'Times New Roman', serif; border-left: 4px solid #09C; padding-left: 8px;"><q>\\1</q></blockquote></p>"""


class AnnotationTask(NamedTuple):
    node_id: str
    tree_id: str
    claim: str
    op: str
    op_text: str
    prev_reply: str #List[Tuple[str, str]]
    user_name: str
    text: str
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
    tree_title = first_node[EXTRA_DATA_FIELD][TITLE_FIELD].strip()[4:].lstrip()
    print(f"Claim char length: {len(tree_title)}\n{tree_title}")
    op: str = first_node[AUTHOR_FIELD]
    op_text = first_node[TEXT_FIELD]
    current_branch_nodes: List[dict] = [first_node]  # Stores the previous nodes in the parsed branch
    current_branch_replies: List[Tuple[str, str]] = [(op, op_text)]  # Stores the previous nodes in the parsed branch

    users_replies_count = Counter()
    tree_nodes = walk_tree(tree, max_depth=MAX_DEPTH + 1)
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
            branch_rerplies = format_branch_replies(current_branch_replies[1:][-1:])
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


def format_quotes(text: str):
    return QUOTE_PATT.sub(BLOCKQUOTE_TEMPLATE, text)


def get_num_quotes(text: str) -> int:
    return len(QUOTE_PATT.findall(text))


def get_quotes_portion(text: str) -> float:
    num_quotes = get_num_quotes(text)
    quotes_size = sum(len(match.group(1)) for match in QUOTE_PATT.finditer(text))
    return quotes_size / (len(text) - (num_quotes * QUOTE_TAGS_SIZE))


def format_reply_text(text: str) -> str:
    return format_quotes(text)


def format_prev_text(text: str):
    global MAX_DISPLAY_TEXT_LEN

    if len(text) <= MAX_DISPLAY_TEXT_LEN:
        return text

    splitted_text = text.split(" ")
    last_token_index = 0
    total_length = 0
    for i, token in enumerate(splitted_text):
        total_length += len(token) + 1
        if total_length >= MAX_DISPLAY_TEXT_LEN:
            last_token_index = i
            break

    shorter_text = " ".join(splitted_text[:last_token_index]) + " ... "
    shorter_text = format_reply_text(shorter_text)
    return shorter_text


def format_branch_replies(branch_replies: List[Tuple[str, str]]) -> str:
    branch_repr = "\n".join((f"{format_prev_text(reply)}\n" for _, reply in branch_replies))
    return branch_repr


def is_relevant_tree(tree: dict) -> bool:
    op = tree[NODE_FIELD][AUTHOR_FIELD]
    if op == "[deleted]":
        return False
    if len(tree[NODE_FIELD][EXTRA_DATA_FIELD][TITLE_FIELD]) > MAX_TITLE_LENGTH:
        return False

    return True


def is_relevant_task(task: AnnotationTask, relevant_users: Set[str]) -> bool:
    if task.user_name not in relevant_users:
        return False
    if get_num_quotes(task.text) > MAX_QUOTES:
        return False
    if get_quotes_portion(task.text) > MAX_QUOTE_PORTION:
        return False
    if len(task.text) > MAX_TEXT_LEN:
        return False
    if task.text == "[removed]":
        return False

    return True


def get_users_with_multiple_tasks(tasks: Iterable[AnnotationTask]):
    counts = Counter((t.user_name for t in tasks))
    return set(map(itemgetter(0), takewhile(lambda t: t[1] >= 2, counts.most_common())))


def filter_tasks(tree: dict, ann_tasks: Iterable[AnnotationTask]) -> List[AnnotationTask]:
    if not is_relevant_tree(tree):
        return []

    op = tree[NODE_FIELD][AUTHOR_FIELD]
    interactions = parse_users_interactions(tree)
    undir_graph = get_core_interactions(interactions, op)
    users = set(undir_graph.nodes())
    if len(users) < MIN_CORE_SIZE:
        return []

    users.discard("[deleted]")
    task_relevancy_func = partial(is_relevant_task, relevant_users=users)
    relevant_tasks = list(filter(task_relevancy_func, ann_tasks))

    # relevant_users = get_users_with_multiple_tasks(relevant_tasks)
    # relevant_tasks = filter(lambda t: t.user_name in relevant_users, relevant_tasks)
    return list(relevant_tasks)


def write_annotation_tasks(p_tasks: Iterable[AnnotationTask], path: str):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',', escapechar='"', quoting=csv.QUOTE_ALL, lineterminator='\r\n')

        # write header
        header = AnnotationTask.get_fields()
        writer.writerow(header)

        for task in p_tasks:
            record = list(task)
            record[7] = format_reply_text(record[7])
            writer.writerow(record)


if __name__ == "__main__":

    labeled_trees_path = sys.argv[1]    # "/home/ron/data/bgu/labeled/61019_notcut_trees.txt"
    outpath = sys.argv[2]    # "/home/ron/data/bgu/stance_annotation/tasks_v1.0.0.csv"

    trees = enumerate(iter_trees(labeled_trees_path))
    deque(islice(trees, 18), maxlen=0)
    trees = islice(trees, 40)
    tasks = []
    for i, tree in trees:
        print(f"tree: {i}")
        tree_tasks = prepare_annotation_tasks_from_tree(tree)
        tree_tasks = filter_tasks(tree, tree_tasks)
        shuffle(tree_tasks)
        tasks.extend(tree_tasks)
        print(f"Num tasks added: {len(tree_tasks)}")
        print(f"Num users added: {len(set([t.user_name for t in tree_tasks]))}")
        print()

    print(f"Total number of tasks: {len(tasks)}")
    print(f"Total number of users: {len(set([(t.tree_id, t.user_name) for t in tasks]))}")
    print(f"Total number of trees: {len(set([t.tree_id for t in tasks]))}")
    # first_tasks = tasks[:180]
    write_annotation_tasks(tasks, outpath)

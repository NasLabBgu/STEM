import csv
from operator import itemgetter

from typing import Sequence, Iterable, List

import tqdm

from conversation import Conversation
from stance_classification.utils import iter_trees_from_lines
from stance_classification.reddit_conversation_parser import CMVConversationReader
from stance_classification.data.text_processing import process_text


def conversation_to_records(conv: Conversation, conv_id : str = 0) -> Iterable[List[str]]:
    records = []
    for depth, node in conv.iter_conversation():
        text = node.data["text"]
        processed_text = process_text(text)
        record = [conv_id, node.node_id, node.parent_id, node.data["text"], processed_text, node.author]
        record = list(map(str, record))
        records.append(record)

    records.sort(key=itemgetter(5))
    return records


def get_conversation_size(conv: Conversation) -> int:
    return len(list(conv.iter_conversation()))


if __name__ == "__main__":
    trees_path = r"C:\Users\ronp\Documents\stance-classification\trees_2.0.txt"
    outpath = "cmv_recsys-project.tsv"

    f = open(outpath, 'w', encoding='utf8')
    writer = csv.writer(f, delimiter='\t', lineterminator='\n')
    output_header = ["conv_id", "message_id", "parent_id", "content", "cleaned_content", "timestamp"]
    writer.writerow(output_header)

    # load trees
    # total_trees = sum(1 for _ in iter_trees_from_lines(trees_path))
    trees = tqdm.tqdm(iter_trees_from_lines(trees_path), total=total_trees)

    # convert trees into conversations
    cmv_reader = CMVConversationReader()
    conversations = map(cmv_reader.parse, trees)

    for i, conv in enumerate(conversations):
        conv_size = get_conversation_size(conv)
        if conv_size < 10:
            continue

        records = conversation_to_records(conv, str(i))
        writer.writerows(records)
        # for record in records:
        #     f.write("\t".join(record))
        #     f.write("\n")



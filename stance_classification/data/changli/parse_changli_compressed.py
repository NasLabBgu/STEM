import csv
from typing import Dict, List, Tuple

import pickle


def load_pickled_dict(path: str) -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_post_label_mapping(path: str) -> Dict[str, int]:
    return load_pickled_dict(path)


def load_post_author_mapping(path: str) -> Dict[str, str]:
    return load_pickled_dict(path)


def load_thread_posts_mapping(path: str) -> Dict[str, List[str]]:
    return load_pickled_dict(path)


def load_post_parent_mapping(path: str) -> Dict[str, Tuple[str, int]]:
    data = load_pickled_dict(path)
    return {post_id: tuple(parent[0]) for post_id, parent in data.items()}


def load_auhtors_index(path: str) -> Dict[str, int]:
    with open(path, 'r') as f:
        splitted_lines = (l.split('\t') for l in f)
        pairs = ((t[1].strip(), int(t[0].strip())) for t in splitted_lines)
        return dict(pairs)


def load_original_post_parent_mapping(path: str) -> Dict[Tuple[int, int], Tuple[str, bool]]:
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        post_parent_mapping = {}
        for record in reader:
            discussion_id = int(record[0])
            post_id = int(record[1])
            parent_id = record[4]
            missing_parent = bool(record[5])
            post_parent_mapping[(discussion_id, post_id)] = (parent_id, missing_parent)

        return post_parent_mapping


def decode_original_post_identification(post_id: str) -> Tuple[str, int, int]:
    topic, numeric_id = post_id.split('.')
    original_discussion_index = int(numeric_id[:-5])
    original_post_index = int(numeric_id[-3:])
    return topic, original_discussion_index, original_post_index


def encode_post_identification(topic: str, discussion_index: int, post_index: int) -> str:
    return f"{topic}.{discussion_index}00{str(post_index).zfill(3)}"


if __name__ == "__main__":
    authors_path = "/home/dev/data/stance/chang-li/data/compressed-4forum/allPostAuthorMap.pickle"
    labels_path = "/home/dev/data/stance/chang-li/data/compressed-4forum/allPostLabelMap.pickle"
    threads_path = "/home/dev/data/stance/chang-li/data/compressed-4forum/allThreadPost.pickle"
    replies_path = "/home/dev/data/stance/chang-li/data/compressed-4forum/allPostLinkMap.pickle"
    authors_index_path = "/home/dev/data/stance/IAC/alternative/fourforums/author.txt"
    topic_index_path = "/home/dev/data/stance/IAC/alternative/fourforums/topic.txt"
    original_posts_path = "/home/dev/data/stance/IAC/alternative/fourforums/post.txt"

    outpath = "/home/dev/data/stance/chang-li/data/4forum/records.csv"

    post_author_mapping = load_post_author_mapping(authors_path)
    post_label_mapping = load_post_label_mapping(labels_path)
    thread_posts_mapping = load_thread_posts_mapping(threads_path)
    post_parent_mapping = load_post_parent_mapping(replies_path)
    authors_index = load_auhtors_index(authors_index_path)
    topics_index = load_auhtors_index(topic_index_path)
    original_post_parent_mapping = load_original_post_parent_mapping(original_posts_path)

    ## build records
    header = ["topic", "discussion_id", "post_id", "author", "parent_id", "agreement", "missing_parent", "stance_label",
              "group_label", "topic_index", "discussion_index", "post_index", "author_index", "parent_index"]
    records = []
    for thread, post_ids in thread_posts_mapping.items():
        root_post_id = post_ids[0]
        for i, post_id in enumerate(post_ids):
            topic, discussion_index, post_index = decode_original_post_identification(post_id)
            topic_index = -1
            author = post_author_mapping[post_id]
            author_index = authors_index[author]
            stance_label = post_label_mapping[post_id]
            group_label = stance_label % 2

            # handle parent
            agreement = None
            parent_id = None
            if post_id not in post_parent_mapping:
                parent_index, missing_parent = original_post_parent_mapping[(discussion_index, post_index)]
                if parent_index != "\\N":
                    parent_id = encode_post_identification(topic, discussion_index, int(parent_index))
            else:
                parent_id, agreement = post_parent_mapping[post_id]
                missing_parent = False
                (_, _, parent_index) = decode_original_post_identification(parent_id)

            record = [
                topic, thread, post_id, author, parent_id, agreement, missing_parent, stance_label, group_label,
                topic_index, discussion_index, post_index, author_index, parent_index
            ]
            records.append(record)

    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

from collections import Counter

from itertools import starmap
from typing import NamedTuple, Iterable

import pandas as pd


class AuthorAnnotations(NamedTuple):
    discussion_id: int
    author_id: int
    topic_id: int
    topic_stance_id_1: int
    votes_1: int
    topic_stance_id_2: int
    votes_2: int
    topic_stance_votes_other: int


class AuthorLabel(NamedTuple):
    discussion_id: int
    author_id: int
    topic_id: int
    stance: int

    @staticmethod
    def from_annotation(annotation: AuthorAnnotations, stance: int = None) -> 'AuthorLabel':
        return AuthorLabel(
            annotation.discussion_id,
            annotation.author_id,
            annotation.topic_id,
            stance
        )


def resolve_author_label(annotation: AuthorAnnotations) -> AuthorLabel:

    stance1, stance2 = annotation.topic_stance_id_1, annotation.topic_stance_id_2
    votes1, votes2 = annotation.votes_1, annotation.votes_2

    # multiple votes only for stance label 1
    if (votes1 >= 3) and (votes2 == 0):
        return AuthorLabel.from_annotation(annotation, stance1)
    # multiple votes only for stance label 2
    if (votes1 == 0) and (votes2 >= 3):
        return AuthorLabel.from_annotation(annotation, stance2)
    # both stance labels got votes, significant difference should be between them
    if (votes1 != 0) and (votes2 != 0):
        max_votes = max(votes1, votes2)
        min_votes = votes1 if max_votes is votes2 else votes2
        if max_votes / min_votes > 2:
            stance = stance1 if max_votes is votes1 else stance2
            return AuthorLabel.from_annotation(annotation, stance)

    # not a significant difference between stance labels - so cannot be resolved.
    return AuthorLabel.from_annotation(annotation, None)


def load_author_labels(path: str) -> Iterable[AuthorLabel]:
    df = pd.read_csv(path, '\t')
    records = df.itertuples(index=False, name=None)
    annotations = starmap(AuthorAnnotations, records)
    labeled_authors = map(resolve_author_label, annotations)
    return labeled_authors


if __name__ == "__main__":
    path = "/home/dev/data/stance/IAC/alternative/fourforums/mturk_author_stance.txt"
    labels = list(load_author_labels(path))
    print(labels)
    topics = {3, 7, 8, 9}
    topic_counts = Counter()
    for i in range(len(labels)):
        if labels[i].stance is None:
            continue
        topic = labels[i].topic_id
        if topic not in topics:
            continue

        topic_counts.update([topic])

    print(topic_counts)

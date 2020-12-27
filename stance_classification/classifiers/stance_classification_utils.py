from typing import Set, Tuple, Dict, Iterable

import csv


def load_labels(path) -> Dict[Tuple[str, str], bool]:
    with open(path, 'r') as labels_f:
        reader = csv.reader(labels_f)
        next(reader) # skip header
        nodes_labels_mapping = {tuple(record[0: 2]): bool(int(record[2])) for record in reader}
        return nodes_labels_mapping


def get_cut_from_nodelist(edges: Iterable[Tuple[str, str]], stance_group: Set[str]) -> Set[Tuple[str, str]]:
    cut = set()
    for edge in edges:
        n1, n2 = edge
        if (n1 in stance_group) ^ (n2 in stance_group):
            cut.add(edge)

    return cut

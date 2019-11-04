from typing import List, Set, Tuple


def get_cut_from_nodelist(edges: List[Tuple[str, str]], stance_group: Set[str]) -> Set[Tuple[str, str]]:
    cut = set()
    for edge in edges:
        n1, n2 = edge
        if (n1 in stance_group) ^ (n2 in stance_group):
            cut.add(edge)

    return cut

from functools import partial
from typing import Set, Tuple, Callable, Any, Dict, Sequence

import pylab
import numpy as np
from networkx.algorithms import bipartite

from stance_classification.draw_utils import OP_COLOR, SUPPORT_COLOR, new_figure, OPPOSE_COLOR, CUT_EDGE_COLOR, \
    NON_CUT_EDGE_COLOR, NODE_LABEL_COLOR, TRUE_SUPPORT_COLOR, UNKNOWN_GT_LABEL, TRUE_OPPOSE_COLOR
from stance_classification.classifiers.base_stance_classifier import BaseStanceClassifier
import networkx as nx

from stance_classification.classifiers.maxcut import max_cut
from stance_classification.classifiers.stance_classification_utils import get_cut_from_nodelist

from picos import SymmetricVariable


class MaxcutStanceClassifier(BaseStanceClassifier):

    def __init__(self, weight_field: str = "weight"):
        self.initialized = False
        self.__weight_field = weight_field

        # input
        self.graph: nx.Graph = None
        self.op: str = None

        # result
        self.cut_value: float = -1
        self.cut: Set[Tuple[str, str]] = None
        self.embeddings: Dict[Any, np.ndarray] = None
        self.supporters: Set[str] = None
        self.complement: Set[str] = None

    def set_input(self, graph: nx.Graph, op: Any):
        self.graph = graph
        self.op = op
        self.initialized = True

    def classify_stance(self):
        cut_value, cut_nodes, embeddings = max_cut(self.graph, weight=self.__weight_field)
        self.cut_value = cut_value
        self.supporters = self.__get_supporters(cut_nodes, self.op)
        self.complement = self.graph.nodes - self.supporters
        self.cut = get_cut_from_nodelist(set(self.graph.edges), set(self.supporters))
        self.embeddings = {n: np.array(v) for n, v in embeddings.items()}

    def get_supporters(self) -> Set[str]:
        return self.supporters

    def get_complement(self) -> Set[str]:
        return self.complement

    def get_cut(self) -> Set[Tuple[str, str]]:
        return self.cut

    def clear(self):
        pass

    def draw(self, layout_func: Callable = None, outpath: str = None, true_labels: Dict[Any, int] = None, show: bool = False):
        leave = [e for e in self.graph.edges if e not in self.cut]
        # Close the old figure and open a new one.
        fig = new_figure()
        # Assign colors based on set membership.

        node_colors = [OP_COLOR if (n == self.op) else (SUPPORT_COLOR if n in self.supporters else OPPOSE_COLOR)
                       for n in self.graph.nodes]

        true_labels = true_labels or {}
        op_label = true_labels.get(self.op)
        gt_colors = []
        for node in self.graph.nodes:
            node_true_label = true_labels.get(node)
            if node_true_label is None:
                gt_colors.append(UNKNOWN_GT_LABEL)
            elif node_true_label == op_label:
                gt_colors.append(TRUE_SUPPORT_COLOR)
            else:
                gt_colors.append(TRUE_OPPOSE_COLOR)

        # Draw the nodes and the edges that are not in the cut.
        if layout_func is None:
            layout_func = partial(nx.spring_layout, seed=1919)

        pos = layout_func(self.graph)
        nodes = nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors)
        nodes.set_edgecolor(gt_colors)
        nodes.set_linewidth([v *2 for v in nodes.get_linewidth()])
        node_labels = {n: str(n) for n in self.graph.nodes}
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_color=NODE_LABEL_COLOR)

        # Draw the edges that are in the cut.
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.cut, edge_color=CUT_EDGE_COLOR)
        labels = {e: f"{self.graph[e[0]][e[1]]['weight']:.2f}" for e in self.cut}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_color=CUT_EDGE_COLOR)

        # Draw the edges that are not in the cut
        nx.draw_networkx_edges(self.graph, pos, edgelist=leave, edge_color=NON_CUT_EDGE_COLOR)
        labels = {e: f"{self.graph[e[0]][e[1]]['weight']:.2f}" for e in leave}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_color=NON_CUT_EDGE_COLOR)

        # Show the relaxation optimum value and the cut capacity.
        sval = sum(self.graph[e[0]][e[1]]['weight'] for e in self.cut)
        fig.suptitle(f"{type(self).__name__}\ncut value: {sval}")

        # Show the figure.
        if outpath is not None:
            pylab.savefig(outpath)

        if show:
            pylab.show()

    def __get_supporters(self,  cut_nodes: Set[Any], op: Any):
        return cut_nodes if op in cut_nodes else self.graph.nodes - cut_nodes


from functools import partial
from typing import Set, Tuple, Callable

import pylab
from networkx.algorithms import bipartite

from stance_classification.draw_utils import OP_COLOR, SUPPORT_COLOR, new_figure, OPPOSE_COLOR
from stance_classification.classifiers.base_stance_classifier import BaseStanceClassifier
import networkx as nx

from stance_classification.classifiers.maxcut import max_cut
from stance_classification.classifiers.stance_classification_utils import get_cut_from_nodelist


class MaxcutStanceClassifier(BaseStanceClassifier):

    def __init__(self):
        self.initialized = False

        # input
        self.graph: nx.Graph = None
        self.op: str = None

        # result
        self.cut: Set[Tuple[str, str]] = None
        self.supporters: Set[str] = None
        self.complement: Set[str] = None

    def set_input(self, graph: nx.Graph):
        self.graph = graph
        self.initialized = True

    def classify_stance(self, op: str, algo='prim'):
        self.op = op

        rval, cut_nodes = max_cut(self.graph)
        supporters = cut_nodes
        opposers = self.graph.nodes - cut_nodes
        if op not in cut_nodes:
            supporters, opposers = opposers, supporters

        self.supporters = supporters
        self.complement = opposers
        self.cut = get_cut_from_nodelist(set(self.graph.edges), set(self.supporters))

    def get_supporters(self) -> Set[str]:
        return self.supporters

    def get_complement(self) -> Set[str]:
        return self.complement

    def get_cut(self) -> Set[Tuple[str, str]]:
        return self.cut

    def clear(self):
        pass

    def draw(self, layout_func: Callable = None, outpath: str = None):
        leave = [e for e in self.graph.edges if e not in self.cut]
        # Close the old figure and open a new one.
        fig = new_figure()
        # Assign colors based on set membership.

        node_colors = [OP_COLOR if (n == self.op) else (SUPPORT_COLOR if n in self.supporters else OPPOSE_COLOR)
                       for n in self.graph.nodes]

        # Draw the nodes and the edges that are not in the cut.
        if layout_func is None:
            layout_func = partial(nx.spring_layout, seed=1919)

        pos = layout_func(self.graph)
        nx.draw_networkx(self.graph, pos, node_color=node_colors, edgelist=leave)
        labels = {e: '{}'.format(self.graph[e[0]][e[1]]['weight']) for e in leave}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)

        # Draw the edges that are in the cut.
        nx.draw_networkx_edges(self.graph, pos, edgelist=self.cut, edge_color='r')
        labels = {e: '{}'.format(self.graph[e[0]][e[1]]['weight']) for e in self.cut}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, font_color='r')

        # Show the relaxation optimum value and the cut capacity.
        sval = sum(self.graph[e[0]][e[1]]['weight'] for e in self.cut)
        fig.suptitle(f"Random Stance Classifier\ncut value: {sval}")

        # Show the figure.
        if outpath is not None:
            pylab.savefig(outpath)

        pylab.show()
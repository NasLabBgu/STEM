from functools import partial
from typing import Tuple, List, Set, Callable
from random import Random
import networkx as nx
import pylab

from stance_classification.draw_utils import new_figure, SUPPORT_COLOR, OPPOSE_COLOR, OP_COLOR
from stance_classification.classifiers.base_stance_classifier import BaseStanceClassifier
from stance_classification.classifiers.stance_classification_utils import get_cut_from_nodelist

DEFAULT_RANDOM_SEED = 1919


class RandomStanceClassifier(BaseStanceClassifier):

    def __init__(self, random_seed: int = None, support_prior: float = 0.5):
        self.initialized = False
        random_seed = DEFAULT_RANDOM_SEED if random_seed is None else random_seed
        self.random: Random = Random(random_seed)

        # input
        self.graph: nx.Graph = None
        self.op: str = None
        self.support_prior: float = support_prior

        # result
        self.cut: Set[Tuple[str, str]] = None
        self.supporters: Set[str] = None
        self.complement: Set[str] = None


    def set_input(self, graph: nx.Graph):
        self.graph = graph
        self.initialized = True

    def classify_stance(self, op: str, support_prior: float = None):
        if not self.initialized:
            raise Exception("Class wasn't initialized properly before calling 'classify_stance' method")

        self.op = op
        if support_prior is not None:
            self.support_prior = support_prior

        is_supporter = partial(self.is_supporter, probability=self.support_prior)
        nodes_mask = [is_supporter() if n != self.op else True for n in self.graph.nodes]
        self.supporters = set()
        self.complement = set()
        [self.supporters.add(n) if supporter else self.complement.add(n)
            for n, supporter in zip(self.graph.nodes, nodes_mask)]

        self.cut = get_cut_from_nodelist(set(self.graph.edges), set(self.supporters))

    def is_supporter(self, probability: float) -> bool:
        return self.random.uniform(0, 1) < probability

    def get_supporters(self) -> Set[str]:
        return self.supporters

    def get_complement(self) -> Set[str]:
        return self.complement

    def get_cut(self) -> Set[Tuple[str, str]]:
        return self.cut

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
        fig.suptitle(f"Random Stance Classifier\nsupport prior: {self.support_prior}\ncut value: {sval}")

        # Show the figure.
        if outpath is not None:
            pylab.savefig(outpath)

        pylab.show()



    def clear(self):
        self.initialized = False
        self.graph = None
        self.op = None
        self.support_prior = None



# Define a plotting helper that closes the old and opens a new figure.
from typing import Dict, List

import networkx as nx
import pylab
from networkx.drawing.nx_agraph import graphviz_layout


OP_COLOR = 'tab:blue'
SUPPORT_COLOR = 'lightgreen'
TRUE_SUPPORT_COLOR = "tab:green"
OPPOSE_COLOR = 'lightcoral'
TRUE_OPPOSE_COLOR = "tab:red"
UNKNOWN_GT_LABEL = "tab:gray"
CUT_EDGE_COLOR = "black"
NON_CUT_EDGE_COLOR = "darkgray"
NODE_LABEL_COLOR = "tab:brown"

fig: pylab.Figure = None


def new_figure() -> pylab.Figure:
    try:
        global fig
        pylab.close(fig)
    except NameError:
        pass
    fig = pylab.figure(figsize=(20, 20), dpi=200)
    fig.gca().axes.get_xaxis().set_ticks([])
    fig.gca().axes.get_yaxis().set_ticks([])
    return fig


def draw_graph(graph: nx.Graph, graphviz=True, weight_field: str = "weight", path: str = None):

    pylab.plt.figure(figsize=(20, 15))
    if graphviz:
        pos = graphviz_layout(graph, prog='fdp')
    else:
        pos = nx.spring_layout(graph, seed=1919)

    edges_width = [max(edge[2][weight_field], 0.1) for edge in graph.edges(data=True)]
    edges_transparency = [edge[2][weight_field] for edge in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, width=edges_width, alpha=0.5)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    # nx.draw_networkx(graph, pos)

    if path is not None:
        pylab.plt.savefig(path)

    pylab.plt.show()


def draw_tree(tree: nx.Graph, path: str = None, title: str = None):
    pylab.plt.figure(figsize=(20, 20), dpi=200)
    pos = nx.nx_pydot.graphviz_layout(tree, prog='dot')

    nx.draw_networkx_edges(tree, pos)
    nx.draw_networkx_nodes(tree, pos, node_color="lightgray")
    labels = {u: u.split("-")[0] for u in tree.nodes}
    nx.draw_networkx_labels(tree, pos, labels=labels)
    # nx.draw_networkx(graph, pos)

    if path is not None:
        pylab.plt.savefig(path)

    if title is not None:
        pylab.plt.title(title)

    pylab.plt.show()





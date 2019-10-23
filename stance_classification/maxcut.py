from typing import Dict, Tuple

import picos as pic
import cvxopt as cvx
import cvxopt.lapack

import networkx as nx
import numpy as np
import pylab


OP_COLOR = 'green'
SUPPORT_COLOR = 'lightgreen'
OPPOSE_COLOR = 'lightblue'


def solve_maxcut(G: nx.Graph):
    """

    :param G: undirected graph
    :return:
    """

    num_nodes = G.number_of_nodes()
    maxcut = pic.Problem()

    # Add the symmetric matrix variable.
    X = maxcut.add_variable('X', (num_nodes, num_nodes), 'symmetric')

    # Retrieve the Laplacian of the graph.
    LL = 1 / 4. * nx.laplacian_matrix(G).todense()
    L = pic.new_param('L', LL)

    # Constrain X to have ones on the diagonal.
    maxcut.add_constraint(pic.tools.diag_vect(X) == 1)

    # Constrain X to be positive semidefinite.
    maxcut.add_constraint(X >> 0)

    # Set the objective.
    maxcut.set_objective('max', L | X)

    # print(maxcut)

    # Solve the problem.
    maxcut.solve(verbose=0, solver='cvxopt')

    # print('bound from the SDP relaxation: {0}'.format(maxcut.obj_value()))

    return G, X, L, num_nodes, maxcut


def find_relaxation(G, X, L, num_nodes, maxcut) -> Tuple[float, set]:
    """
    :param X: Positive Semidefinite matrix
    :param L:  Laplacian matix of the graph
    :param num_nodes:
    :param maxcut:
    :return:
    """
    ### Perform the random relaxation
    # Use a fixed RNG seed so the result is reproducable.
    cvx.setseed(1919)

    # Perform a Cholesky factorization.
    V = X.value
    cvxopt.lapack.potrf(V)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            V[i, j] = 0

    # Do up to 100 projections. Stop if we are within a factor 0.878 of the SDP
    # optimal value.
    count = 0
    obj_sdp = maxcut.obj_value()
    obj = 0
    x_cut: cvx.matrix = None
    while (count < 100 or obj < 0.878 * obj_sdp):
        r = cvx.normal(num_nodes, 1)
        x = cvx.matrix(np.sign(V * r))
        o = (x.T * L * x).value
        if o > obj:
            x_cut = x
            obj = o
        count += 1
    x = x_cut

    # Extract the cut and the seperated node sets.
    indexed_nodes = list(G.nodes)
    cut_nodes = [indexed_nodes[i] for i in range(num_nodes) if x[i] < 0]
    return maxcut.obj_value(), cut_nodes


def max_cut(G: nx.Graph) -> Tuple[float, set]:
    """
    :param G: an undirected graph
    :param weights:
    :return:
    """
    if G.number_of_nodes() <= 1:
        return 0, G.nodes()

    G, X, L, num_nodes, maxcut = solve_maxcut(G)
    relxation_value, cut_nodes = find_relaxation(G, X, L, num_nodes, maxcut)
    return relxation_value, cut_nodes


def draw_maxcut(graph: nx.Graph, cut_nodes: set, relaxation_value: float, op: str = None, outpath=None):

    cut = set([(i, j) for i, j in graph.edges if (i in cut_nodes) ^ (j in cut_nodes)])
    leave = [e for e in graph.edges if e not in cut]

    ### Drawing the cut
    # Close the old figure and open a new one.
    new_figure()
    # Assign colors based on set membership.


    cut_color = SUPPORT_COLOR  if op in cut_nodes else OPPOSE_COLOR
    noncut_color = OPPOSE_COLOR if cut_color == SUPPORT_COLOR else SUPPORT_COLOR

    node_colors = [OP_COLOR if (n == op) else (cut_color if n in cut_nodes else noncut_color) for n in graph.nodes]

    # draw op node with different color
    if op is not None:
        op_color = 'green' if op in cut_nodes else 'blue'

    # Draw the nodes and the edges that are not in the cut.
    pos = nx.spring_layout(graph, seed=1919)
    nx.draw_networkx(graph, pos, node_color=node_colors, edgelist=leave)
    labels = {e: '{}'.format(graph[e[0]][e[1]]['weight']) for e in leave}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    # Draw the edges that are in the cut.
    nx.draw_networkx_edges(graph, pos, edgelist=cut, edge_color='r')
    labels = {e: '{}'.format(graph[e[0]][e[1]]['weight']) for e in cut}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, font_color='r')

    # Show the relaxation optimum value and the cut capacity.
    sval = sum(graph[e[0]][e[1]]['weight'] for e in cut)
    fig.suptitle(
        'SDP relaxation value: {0:.1f}\nCut value: {1:.1f} = {2:.3f}×{0:.1f}'
            .format(relaxation_value, sval, sval / relaxation_value), fontsize=16, y=0.97)

    # Show the figure.
    if outpath is not None:
        pylab.savefig(outpath)

    pylab.show()


# Define a plotting helper that closes the old and opens a new figure.
def new_figure():
    try:
        global fig
        pylab.close(fig)
    except NameError:
        pass
    fig = pylab.figure(figsize=(20, 15))
    fig.gca().axes.get_xaxis().set_ticks([])
    fig.gca().axes.get_yaxis().set_ticks([])

if __name__ == "__main__":
    pass
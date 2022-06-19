import json
import random
from typing import Iterable, Dict, Any, List, Tuple

import pandas as pd
import igraph
# from igraph import Graph as , EdgeSeq

from tqdm.auto import tqdm
import numpy as np
import textwrap

import plotly.graph_objects as go
import plotly.express as px

from conversation import Conversation, ConversationNode
from conversation.parse import DataFrameConversationReader

# COLORS = px.colors.qualitative.Plotly

s="""
    aliceblue, antiquewhite, aqua, aquamarine, azure,
    beige, bisque, black, blanchedalmond, blue,
    blueviolet, brown, burlywood, cadetblue,
    chartreuse, chocolate, coral, cornflowerblue,
    cornsilk, crimson, cyan, darkblue, darkcyan,
    darkgoldenrod, darkgray, darkgrey, darkgreen,
    darkkhaki, darkmagenta, darkolivegreen, darkorange,
    darkorchid, darkred, darksalmon, darkseagreen,
    darkslateblue, darkslategray, darkslategrey,
    darkturquoise, darkviolet, deeppink, deepskyblue,
    dimgray, dimgrey, dodgerblue, firebrick,
    floralwhite, forestgreen, fuchsia, gainsboro,
    ghostwhite, gold, goldenrod, gray, grey, green,
    greenyellow, honeydew, hotpink, indianred, indigo,
    ivory, khaki, lavender, lavenderblush, lawngreen,
    lemonchiffon, lightblue, lightcoral, lightcyan,
    lightgoldenrodyellow, lightgray, lightgrey,
    lightgreen, lightpink, lightsalmon, lightseagreen,
    lightskyblue, lightslategray, lightslategrey,
    lightsteelblue, lightyellow, lime, limegreen,
    linen, magenta, maroon, mediumaquamarine,
    mediumblue, mediumorchid, mediumpurple,
    mediumseagreen, mediumslateblue, mediumspringgreen,
    mediumturquoise, mediumvioletred, midnightblue,
    mintcream, mistyrose, moccasin, navajowhite, navy,
    oldlace, olive, olivedrab, orange, orangered,
    orchid, palegoldenrod, palegreen, paleturquoise,
    palevioletred, papayawhip, peachpuff, peru, pink,
    plum, powderblue, purple, red, rosybrown,
    royalblue, saddlebrown, salmon, sandybrown,
    seagreen, seashell, sienna, silver, skyblue,
    slateblue, slategray, slategrey, snow, springgreen,
    steelblue, tan, teal, thistle, tomato, turquoise,
    violet, wheat, white, whitesmoke, yellow,
    yellowgreen
"""
COLORS = list(map(str.strip, s.split(',')))

FIELDS_MAPPING = {
    "node_id": "post_id",
    "author": "author",
    "timestamp": "timestamp",
    "parent_id": "parent_post_id"
}


def load_conversations_from_dataframe(path: str) -> Iterable[Conversation]:
    df = pd.read_csv(path, low_memory=False)
    parser = DataFrameConversationReader(FIELDS_MAPPING, conversation_id_column="conversation_id")
    groups = df.groupby("conversation_id")
    for cid, raw_conversation in tqdm(groups, total=groups.ngroups):
        yield parser.parse(raw_conversation, conversation_id=cid)


def get_author_labels(conv: Conversation) -> Dict[Any, int]:
    labels = {}
    for depth, node in conv.iter_conversation():
        label = node.data["author_label"]
        if isinstance(label, str):
            print(node)
        if label >= 0:
            labels[node.author] = label

    return labels


def is_relevant_conversation(conv: Conversation) -> bool:
    return bool(get_author_labels(conv))


def build_tree(conv: Conversation) -> Tuple[igraph.Graph, Dict[ConversationNode, int]]:
    _, nodes = zip(*list(conv.iter_conversation()))
    nodes_indices = {node.node_id: i for i, node in enumerate(nodes)}
    edges = [(nodes_indices[n.node_id], nodes_indices[n.parent_id]) for n in nodes if n.parent_id is not None]
    g = igraph.Graph(n=len(nodes), edges=edges)
    return g, nodes_indices


def customwrap(s: str, min_width: int = 60, extra: int = 20) -> str:
    if len(s) < (min_width + extra):
        return s

    width = max(min_width, int(3 * np.sqrt(len(s))))
    s = s.replace("\n", "\n\n")
    return "<br>".join(textwrap.wrap(s, width=width))


def visualize_discussion_tree(conv: Conversation):
    tree, nodes_index_map = build_tree(conv)
    n_nodes = tree.vcount()
    layout = tree.layout("rt", root=nodes_index_map[conv.root.node_id])
    position = {i: layout[i] for i in range(n_nodes)}

    twice_max_y = 2 * max(y for _, y in position.values())
    Xn = [position[k][0] for k in range(n_nodes)]
    Yn = [twice_max_y - position[k][1] for k in range(n_nodes)]
    Xe, Ye = [], []
    for edge in tree.es:
        s, t = edge.tuple
        sx, sy = position[s]
        tx, ty = position[t]
        Xe.extend((sx, tx, None))
        Ye.extend((twice_max_y - sy, twice_max_y - ty, None))

    labels = [-1 for _ in range(n_nodes)]
    texts = [None for _ in range(n_nodes)]
    ids = [None for _ in range(n_nodes)]
    authors = [None for _ in range(n_nodes)]
    timestamps = [None for _ in range(n_nodes)]
    for _, node in conv.iter_conversation():
        node_index = nodes_index_map[node.node_id]
        ids[node_index] = node.node_id
        authors[node_index] = node.author
        timestamps[node_index] = node.node_data.timestamp
        texts[node_index] = customwrap(node.node_data.data["text"])

    r = random.Random(x=1919)
    authors_index = {a: i for i, a in enumerate(set(authors))}
    sampled_colors = r.sample(COLORS, len(authors_index))
    author_colors = {a: sampled_colors[i] for a, i in authors_index.items()}
    node_colors = [author_colors[authors[i]] for i in range(n_nodes)]
    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    ## visualize
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Xe,
                             y=Ye,
                             mode='lines',
                             line=dict(color='rgb(210,210,210)', width=1),
                             hoverinfo='none'
                             ))
    fig.add_trace(go.Scatter(x=Xn,
                             y=Yn,
                             mode='markers',
                             name="Nodes",
                             marker=dict(symbol='circle-dot',
                                         size=18,
                                         color=node_colors,  # '#DB4551',
                                         line=dict(color='rgb(50,50,50)', width=1)
                                         ),
                             customdata=list(zip(ids, authors, timestamps, texts)),
                             opacity=0.8,
                             hovertemplate='<b>ID</b>: %{customdata[0]}<br><b>Author</b>: %{customdata[1]}<br><b>Timestamp</b>: %{customdata[2]}<br><b>Text</b>: %{customdata[3]}',
                             ))

    annotations = []
    for i in range(n_nodes):
        annotations.append(
            dict(
                text=authors[i],  # or replace labels with a different list for the text within the circle
                x=position[i][0], y=twice_max_y - position[i][1],
                xref='x1', yref='y1',
                font=dict(color='rgb(250,250,250)', size=10),
                showarrow=False)
        )

    fig.update_layout(title=f"Discussion Tree - {conv.id} - {conv.root.data['title']}",
                      annotations=annotations,
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='rgb(248,248,248)'
                      )

    fig.show()


if __name__ == "__main__":
    path = "../data/fourforums/4forums-v4.0.0.csv"
    target_conv_id = 173

    convs = iter(load_conversations_from_dataframe(path))
    conv = next(convs)
    conv_id = None
    while conv.id != target_conv_id:
        conv = next(convs)

    visualize_discussion_tree(conv)

    # convs = list(filter(is_relevant_conversation, convs))



# Define a plotting helper that closes the old and opens a new figure.
import pylab


OP_COLOR = 'green'
SUPPORT_COLOR = 'lightgreen'
OPPOSE_COLOR = 'lightblue'

fig: pylab.Figure = None


def new_figure() -> pylab.Figure:
    try:
        global fig
        pylab.close(fig)
    except NameError:
        pass
    fig = pylab.figure(figsize=(20, 15))
    fig.gca().axes.get_xaxis().set_ticks([])
    fig.gca().axes.get_yaxis().set_ticks([])
    return fig

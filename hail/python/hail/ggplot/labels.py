from .geoms import FigureAttribute


class Labels(FigureAttribute):
    def __init__(self, title=None, xlabel=None, ylabel=None, **kwargs):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def apply_to_fig(self, fig_so_far):
        layout_updates = {}
        if self.title is not None:
            layout_updates["title"] = self.title
        if self.xlabel is not None:
            layout_updates["xaxis_title"] = self.xlabel
        if self.ylabel is not None:
            layout_updates["yaxis_title"] = self.ylabel

        fig_so_far.update_layout(**layout_updates)

    def merge(self, other):
        new_title = other.title if other.title is not None else self.title
        new_xlabel = other.xlabel if other.xlabel is not None else self.xlabel
        new_ylabel = other.ylabel if other.ylabel is not None else self.ylabel

        return Labels(title=new_title, xlabel=new_xlabel, ylabel=new_ylabel)


def ggtitle(label):
    """Sets the title of a plot.

    Parameters
    ----------
    label : :class:`str`
        The desired title of the plot.

    Returns
    -------
    :class:`.FigureAttribute`
        Label object to change the title.
    """
    return Labels(title=label)


def xlab(label):
    """Sets the x-axis label of a plot.

    Parameters
    ----------
    label : :class:`str`
        The desired x-axis label of the plot.

    Returns
    -------
    :class:`.FigureAttribute`
        Label object to change the x-axis label.
    """
    return Labels(xlabel=label)


def ylab(label):
    """Sets the y-axis label of a plot.

    Parameters
    ----------
    label : :class:`str`
        The desired y-axis label of the plot.

    Returns
    -------
    :class:`.FigureAttribute`
        Label object to change the y-axis label.
    """
    return Labels(ylabel=label)

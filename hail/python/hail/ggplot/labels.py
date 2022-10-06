from .geoms import FigureAttribute


class Labels(FigureAttribute):
    def __init__(self, title=None, xlabel=None, ylabel=None, group_labels={}, **kwargs):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.group_labels = group_labels

    def apply_to_fig(self, fig_so_far):
        layout_updates = {}
        if self.title is not None:
            layout_updates["title"] = self.title
        if self.xlabel is not None:
            layout_updates["xaxis_title"] = self.xlabel
        if self.ylabel is not None:
            layout_updates["yaxis_title"] = self.ylabel

        fig_so_far.update_layout(**layout_updates)

        for legend_group, label in self.group_labels.items():
            fig_so_far.update_traces({"legendgrouptitle_text": label}, {"legendgroup": legend_group})

    def merge(self, other):
        new_title = other.title if other.title is not None else self.title
        new_xlabel = other.xlabel if other.xlabel is not None else self.xlabel
        new_ylabel = other.ylabel if other.ylabel is not None else self.ylabel
        new_group_labels = {**self.group_labels, **other.group_labels}

        return Labels(title=new_title, xlabel=new_xlabel, ylabel=new_ylabel, group_labels=new_group_labels)


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


def labs(**group_labels):
    """Sets the labels for the legend groups of a plot.

    Examples
    --------

    Create a scatterplot and label the legend groups according to their field names:

    >>> ht = hl.utils.range_table(10)
    >>> ht = ht.annotate(squared=ht.idx ** 2)
    >>> ht = ht.annotate(even=hl.if_else(ht.idx % 2 == 0, "yes", "no"))
    >>> ht = ht.annotate(threeven=hl.if_else(ht.idx % 3 == 0, "good", "bad"))
    >>> fig = (
    ...     hl.ggplot.ggplot(ht, hl.ggplot.aes(x=ht.idx, y=ht.squared))
    ...     + hl.ggplot.geom_point(hl.ggplot.aes(color=ht.even, shape=ht.threeven))
    ...     + hl.ggplot.labs(color="Even", shape="Threeven")
    ... )

    Parameters
    ----------
    group_labels:
        Map names of plotly ``legendgroup``s to the desired replacement labels.

    Returns
    -------
    :class:`.FigureAttribute`
        Label object to change the legend group labels.
    """
    return Labels(group_labels=group_labels)

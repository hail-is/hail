from .geoms import FigureAttribute


class CoordCartesian(FigureAttribute):
    def __init__(self, xlim, ylim):
        self.xlim = xlim
        self.ylim = ylim

    def apply_to_fig(self, fig_so_far):
        if self.xlim is not None:
            fig_so_far.update_xaxes(range=list(self.xlim))
        if self.ylim is not None:
            fig_so_far.update_yaxes(range=list(self.ylim))


def coord_cartesian(xlim=None, ylim=None):
    """Set the boundaries of the plot.

    Parameters
    ----------
    xlim : :obj:`tuple` with two int
        The minimum and maximum x value to show on the plot.
    ylim : :obj:`tuple` with two int
        The minimum and maximum y value to show on the plot.

    Returns
    -------
    :class:`.FigureAttribute`
        The coordinate attribute to be applied.

    """
    return CoordCartesian(xlim, ylim)

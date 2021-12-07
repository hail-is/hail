from .geoms import FigureAttribute


class Labels(FigureAttribute):
    def __init__(self, title=None, xlabel=None, ylabel=None, **kwargs):
        self.title = title

    def apply_to_fig(self, fig_so_far):
        if self.title is not None:
            fig_so_far.update_layout(title=self.title)

    def merge(self, other):
        new_title = other.title if other.title is not None else self.title

        return Labels(title=new_title)


def ggtitle(label):
    return Labels(title=label)

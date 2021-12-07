from .geoms import FigureAttribute


class Labels(FigureAttribute):
    def __init__(self, title=None):
        self.title = title

    def apply_to_fig(self, parent, collected, mapping_field_name, fig_so_far):
        if self.title is not None:
            fig_so_far.update_layout(title=self.title)


def ggtitle(label):
    Labels(title=label)

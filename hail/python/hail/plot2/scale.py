import abc
from .geoms import FigureAttribute


class Scale(FigureAttribute):
    def __init__(self, aesthetic_name):
        self.aesthetic_name = aesthetic_name

    @abc.abstractmethod
    def transform_data(self, field_expr):
        pass


class PositionScale(Scale):
    def __init__(self, aesthetic_name, name, breaks, labels):
        super().__init__(aesthetic_name)
        self.name = name
        self.breaks = breaks
        self.labels = labels

    def update_axis(self, fig):
        if self.aesthetic_name == "x":
            return fig.update_xaxes
        elif self.aesthetic_name == "y":
            return fig.update_yaxes

    # What else do discrete and continuous scales have in common?
    def apply_to_fig(self, parent, fig_so_far):
        if self.name is not None:
            self.update_axis(fig_so_far)(title=self.name)

        if self.breaks is not None:
            self.update_axis(fig_so_far)(tickvals=self.breaks)

        if self.labels is not None:
            self.update_axis(fig_so_far)(ticktext=self.labels)


class PositionScaleGenomic(PositionScale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name, None, None, None)

    def apply_to_fig(self, parent, fig_so_far):
        ref_genome = parent.aes["x"].dtype.reference_genome
        contig_offsets = dict(list(ref_genome._global_positions_dict.items())[:24])
        breaks = list(contig_offsets.values())
        labels = list(contig_offsets.keys())
        self.update_axis(fig_so_far)(tickvals=breaks, ticktext=labels)

    def transform_data(self, field_expr):
        return field_expr.global_position()


class PositionScaleContinuous(PositionScale):

    def __init__(self, axis=None, name=None, breaks=None, labels=None, transformation="identity"):
        super().__init__(axis, name, breaks, labels)
        self.transformation = transformation

    def apply_to_fig(self, parent, fig_so_far):
        super().apply_to_fig(parent, fig_so_far)
        if self.transformation == "identity":
            pass
        elif self.transformation == "log10":
            self.update_axis(fig_so_far)(type="log")
        elif self.transformation == "reverse":
            self.update_axis(fig_so_far)(autorange="reversed")
        else:
            raise ValueError("Unrecognized transformation")

    def transform_data(self, field_expr):
        return field_expr


def scale_x_log10():
    return PositionScaleContinuous("x", transformation="log10")


def scale_y_log10():
    return PositionScaleContinuous("y", transformation="log10")


def scale_x_reverse():
    return PositionScaleContinuous("x", transformation="reverse")


def scale_y_reverse():
    return PositionScaleContinuous("y", transformation="reverse")


def scale_x_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return PositionScaleContinuous("x", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_y_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return PositionScaleContinuous("y", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_x_genomic(name=None):
    return PositionScaleGenomic("x", name=name)

from .geoms import FigureAttribute


class Scale(FigureAttribute):
    def __init__(self, axis, name, breaks, labels):
        assert axis in ["x", "y"]
        self.name = name
        self.axis = axis
        self.breaks = breaks
        self.labels = labels

    def update_axis(self, fig):
        if self.axis == "x":
            return fig.update_xaxes
        elif self.axis == "y":
            return fig.update_yaxes

    # What else do discrete and continuous scales have in common?
    def apply_to_fig(self, fig_so_far):
        if self.name is not None:
            self.update_axis(fig_so_far)(title=self.name)

        if self.breaks is not None:
            self.update_axis(fig_so_far)(tickvals=self.breaks)

        if self.labels is not None:
            self.update_axis(fig_so_far)(ticktext=self.labels)


class ScaleContinuous(Scale):

    def __init__(self, axis=None, name=None, breaks=None, labels=None, transformation="identity"):
        super().__init__(axis, name, breaks, labels)
        self.transformation = transformation

    def apply_to_fig(self, fig_so_far):
        super().apply_to_fig(fig_so_far)
        if self.transformation == "identity":
            pass
        elif self.transformation == "log10":
            self.update_axis(fig_so_far)(type="log")
        elif self.transformation == "reverse":
            self.update_axis(fig_so_far)(autorange="reversed")
        else:
            raise ValueError("Unrecognized transformation")


def scale_x_log10():
    return ScaleContinuous("x", transformation="log10")


def scale_y_log10():
    return ScaleContinuous("y", transformation="log10")


def scale_x_reverse():
    return ScaleContinuous("x", transformation="reverse")


def scale_y_reverse():
    return ScaleContinuous("y", transformation="reverse")


def scale_x_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return ScaleContinuous("x", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_y_continuous(name=None, breaks=None, labels=None, trans="identity"):
    return ScaleContinuous("y", name=name, breaks=breaks, labels=labels, transformation=trans)

import abc
from .geoms import FigureAttribute

from hail.context import get_reference

from .utils import categorical_strings_to_colors, continuous_nums_to_colors


class Scale(FigureAttribute):
    def __init__(self, aesthetic_name):
        self.aesthetic_name = aesthetic_name

    @abc.abstractmethod
    def transform_data(self, field_expr):
        pass

    def transform_data_local(self, data, parent):
        return data

    @abc.abstractmethod
    def is_discrete(self):
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
    def __init__(self, aesthetic_name, reference_genome, name=None):
        super().__init__(aesthetic_name, name, None, None)

        if isinstance(reference_genome, str):
            reference_genome = get_reference(reference_genome)
        self.reference_genome = reference_genome

    def apply_to_fig(self, parent, fig_so_far):
        contig_offsets = dict(list(self.reference_genome._global_positions_dict.items())[:24])
        breaks = list(contig_offsets.values())
        labels = list(contig_offsets.keys())
        self.update_axis(fig_so_far)(tickvals=breaks, ticktext=labels)

    def transform_data(self, field_expr):
        return field_expr.global_position()

    def is_discrete(self):
        return False


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
            raise ValueError(f"Unrecognized transformation {self.transformation}")

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return False


class PositionScaleDiscrete(PositionScale):
    def __init__(self, axis=None, name=None, breaks=None, labels=None):
        super().__init__(axis, name, breaks, labels)

    def apply_to_fig(self, parent, fig_so_far):
        super().apply_to_fig(parent, fig_so_far)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True


class ScaleContinuous(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return False


class ScaleDiscrete(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True


class ScaleColorDiscrete(ScaleDiscrete):
    def transform_data_local(self, data, parent):
        categorical_strings = set([element[self.aesthetic_name] for element in data])
        unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent)

        updated_data = []
        for category in categorical_strings:
            for data_entry in data:
                if data_entry[self.aesthetic_name] == category:
                    annotate_args = {
                        self.aesthetic_name: unique_color_mapping[category],
                        "color_legend": category
                    }
                    updated_data.append(data_entry.annotate(**annotate_args))
        return updated_data


class ScaleColorContinuous(ScaleContinuous):
    def transform_data_local(self, data, parent):
        color_list = [element[self.aesthetic_name] for element in data]
        color_mapping = continuous_nums_to_colors(color_list, parent.continuous_color_scale)
        updated_data = []
        for data_idx, data_entry in enumerate(data):
            annotate_args = {
                self.aesthetic_name: color_mapping[data_idx],
                "color_legend": data_entry[self.aesthetic_name]
            }
            updated_data.append(data_entry.annotate(**annotate_args))
        return updated_data


# Legend names messed up for scale color identity
class ScaleColorDiscreteIdentity(ScaleDiscrete):
    def transform_data_local(self, data, parent):
        return data


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


def scale_x_discrete(name=None, breaks=None, labels=None):
    return PositionScaleDiscrete("x", name=name, breaks=breaks, labels=labels)


def scale_y_discrete(name=None, breaks=None, labels=None):
    return PositionScaleDiscrete("y", name=name, breaks=breaks, labels=labels)


def scale_x_genomic(reference_genome, name=None):
    return PositionScaleGenomic("x", reference_genome, name=name)


def scale_color_discrete():
    return ScaleColorDiscrete("color")


def scale_color_continuous():
    return ScaleColorContinuous("color")


def scale_color_identity():
    return ScaleColorDiscreteIdentity("color")


def scale_fill_discrete():
    return ScaleColorDiscrete("fill")


def scale_fill_continuous():
    return ScaleColorContinuous("fill")


def scale_fill_identity():
    return ScaleColorDiscreteIdentity("fill")

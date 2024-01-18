import abc
from collections.abc import Mapping

import plotly
import plotly.express as px

from hail.context import get_reference
from hail.expr import tstr

from .geoms import FigureAttribute
from .utils import continuous_nums_to_colors, is_continuous_type, is_discrete_type


class Scale(FigureAttribute):
    def __init__(self, aesthetic_name):
        self.aesthetic_name = aesthetic_name

    @abc.abstractmethod
    def transform_data(self, field_expr):
        pass

    def create_local_transformer(self, groups_of_dfs):
        return lambda x: x

    @abc.abstractmethod
    def is_discrete(self):
        pass

    @abc.abstractmethod
    def is_continuous(self):
        pass

    def valid_dtype(self, dtype):
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

    def valid_dtype(self, dtype):
        return True


class PositionScaleGenomic(PositionScale):
    def __init__(self, aesthetic_name, reference_genome, name=None):
        super().__init__(aesthetic_name, name, None, None)

        if isinstance(reference_genome, str):
            reference_genome = get_reference(reference_genome)
        self.reference_genome = reference_genome

    def apply_to_fig(self, parent, fig_so_far):
        contig_offsets = dict(list(self.reference_genome.global_positions_dict.items())[:24])
        breaks = list(contig_offsets.values())
        labels = list(contig_offsets.keys())
        self.update_axis(fig_so_far)(tickvals=breaks, ticktext=labels)

    def transform_data(self, field_expr):
        return field_expr.global_position()

    def is_discrete(self):
        return False

    def is_continuous(self):
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

    def is_continuous(self):
        return True


class PositionScaleDiscrete(PositionScale):
    def __init__(self, axis=None, name=None, breaks=None, labels=None):
        super().__init__(axis, name, breaks, labels)

    def apply_to_fig(self, parent, fig_so_far):
        super().apply_to_fig(parent, fig_so_far)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False


class ScaleContinuous(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return False

    def is_continuous(self):
        return True

    def valid_dtype(self, dtype):
        return is_continuous_type(dtype)


class ScaleDiscrete(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def get_values(self, categories):
        return None

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False

    def valid_dtype(self, dtype):
        return is_discrete_type(dtype)

    def create_local_transformer(self, groups_of_dfs):
        categories = set()
        for group_of_dfs in groups_of_dfs:
            for df in group_of_dfs:
                if self.aesthetic_name in df.attrs:
                    categories.add(df.attrs[self.aesthetic_name])

        values = self.get_values(categories)

        if values is None:
            return super().create_local_transformer(groups_of_dfs)
        elif isinstance(values, Mapping):
            mapping = values
        elif isinstance(values, list):
            if len(categories) > len(values):
                raise ValueError(
                    f"Not enough scale values specified. Found {len(categories)} "
                    f"distinct categories in {categories} and only {len(values)} "
                    f"scale values were provided in {values}."
                )
            mapping = dict(zip(categories, values))
        else:
            raise TypeError(
                "Expected scale values to be a Mapping or list, but received a(n) " f"{type(values)}: {values}."
            )

        def transform(df):
            df.attrs[f"{self.aesthetic_name}_legend"] = df.attrs[self.aesthetic_name]
            df.attrs[self.aesthetic_name] = mapping[df.attrs[self.aesthetic_name]]
            return df

        return transform


class ScaleDiscreteManual(ScaleDiscrete):
    def __init__(self, aesthetic_name, values):
        super().__init__(aesthetic_name)
        self.values = values

    def get_values(self, categories):
        return self.values


class ScaleColorContinuous(ScaleContinuous):
    def create_local_transformer(self, groups_of_dfs):
        overall_min = None
        overall_max = None
        for group_of_dfs in groups_of_dfs:
            for df in group_of_dfs:
                if self.aesthetic_name in df.columns:
                    series = df[self.aesthetic_name]
                    series_min = series.min()
                    series_max = series.max()
                    if overall_min is None:
                        overall_min = series_min
                    else:
                        overall_min = min(series_min, overall_min)

                    if overall_max is None:
                        overall_max = series_max
                    else:
                        overall_max = max(series_max, overall_max)

        color_mapping = continuous_nums_to_colors(overall_min, overall_max, plotly.colors.sequential.Viridis)

        def transform(df):
            df[self.aesthetic_name] = df[self.aesthetic_name].map(lambda i: color_mapping(i))
            return df

        return transform


class ScaleColorHue(ScaleDiscrete):
    def get_values(self, categories):
        num_categories = len(categories)
        step = 1.0 / num_categories
        interpolation_values = [step * i for i in range(num_categories)]
        hsv_scale = px.colors.get_colorscale("HSV")
        return px.colors.sample_colorscale(hsv_scale, interpolation_values)


class ScaleShapeAuto(ScaleDiscrete):
    def get_values(self, categories):
        return [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "triangle-left",
            "triangle-right",
            "triangle-ne",
            "triangle-se",
            "triangle-sw",
            "triangle-nw",
            "pentagon",
            "hexagon",
            "hexagon2",
            "octagon",
            "star",
            "hexagram",
            "star-triangle-up",
            "star-triangle-down",
            "star-square",
            "star-diamond",
            "diamond-tall",
            "diamond-wide",
            "hourglass",
            "bowtie",
            "circle-cross",
            "circle-x",
            "square-cross",
            "square-x",
            "diamond-cross",
            "diamond-x",
            "cross-thin",
            "x-thin",
            "asterisk",
            "hash",
            "y-up",
            "y-down",
            "y-left",
            "y-right",
            "line-ew",
            "line-ns",
            "line-ne",
            "line-nw",
            "arrow-up",
            "arrow-down",
            "arrow-left",
            "arrow-right",
            "arrow-bar-up",
            "arrow-bar-down",
            "arrow-bar-left",
            "arrow-bar-right",
        ]


class ScaleColorContinuousIdentity(ScaleContinuous):
    def valid_dtype(self, dtype):
        return dtype == tstr


def scale_x_log10(name=None):
    """Transforms x axis to be log base 10 scaled.

    Parameters
    ----------
    name: :class:`str`
        The label to show on x-axis

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("x", name=name, transformation="log10")


def scale_y_log10(name=None):
    """Transforms y-axis to be log base 10 scaled.

    Parameters
    ----------
    name: :class:`str`
        The label to show on y-axis

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("y", name=name, transformation="log10")


def scale_x_reverse(name=None):
    """Transforms x-axis to be vertically reversed.

    Parameters
    ----------
    name: :class:`str`
        The label to show on x-axis

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("x", name=name, transformation="reverse")


def scale_y_reverse(name=None):
    """Transforms y-axis to be vertically reversed.

    Parameters
    ----------
    name: :class:`str`
        The label to show on y-axis

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("y", name=name, transformation="reverse")


def scale_x_continuous(name=None, breaks=None, labels=None, trans="identity"):
    """The default continuous x scale.

    Parameters
    ----------
    name: :class:`str`
        The label to show on x-axis
    breaks: :class:`list` of :class:`float`
        The locations to draw ticks on the x-axis.
    labels: :class:`list` of :class:`str`
        The labels of the ticks on the axis.
    trans: :class:`str`
        The transformation to apply to the x-axis. Supports "identity", "reverse", "log10".

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("x", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_y_continuous(name=None, breaks=None, labels=None, trans="identity"):
    """The default continuous y scale.

    Parameters
    ----------
    name: :class:`str`
        The label to show on y-axis
    breaks: :class:`list` of :class:`float`
        The locations to draw ticks on the y-axis.
    labels: :class:`list` of :class:`str`
        The labels of the ticks on the axis.
    trans: :class:`str`
        The transformation to apply to the y-axis. Supports "identity", "reverse", "log10".

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleContinuous("y", name=name, breaks=breaks, labels=labels, transformation=trans)


def scale_x_discrete(name=None, breaks=None, labels=None):
    """The default discrete x scale.

    Parameters
    ----------
    name: :class:`str`
        The label to show on x-axis
    breaks: :class:`list` of :class:`str`
        The locations to draw ticks on the x-axis.
    labels: :class:`list` of :class:`str`
        The labels of the ticks on the axis.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleDiscrete("x", name=name, breaks=breaks, labels=labels)


def scale_y_discrete(name=None, breaks=None, labels=None):
    """The default discrete y scale.

    Parameters
    ----------
    name: :class:`str`
        The label to show on y-axis
    breaks: :class:`list` of :class:`str`
        The locations to draw ticks on the y-axis.
    labels: :class:`list` of :class:`str`
        The labels of the ticks on the axis.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleDiscrete("y", name=name, breaks=breaks, labels=labels)


def scale_x_genomic(reference_genome, name=None):
    """The default genomic x scale. This is used when the ``x`` aesthetic corresponds to a :class:`.LocusExpression`.

    Parameters
    ----------
    reference_genome:
        The reference genome being used.
    name: :class:`str`
        The label to show on y-axis

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return PositionScaleGenomic("x", reference_genome, name=name)


def scale_color_discrete():
    """The default discrete color scale. This maps each discrete value to a color. Equivalent to scale_color_hue.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return scale_color_hue()


def scale_color_hue():
    """Map discrete colors to evenly placed positions around the color wheel.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.

    """
    return ScaleColorHue("color")


def scale_color_continuous():
    """The default continuous color scale. This linearly interpolates colors between the min and max observed values.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorContinuous("color")


def scale_color_identity():
    """A color scale that assumes the expression specified in the ``color`` aesthetic can be used as a color.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorContinuousIdentity("color")


def scale_color_manual(*, values):
    """A color scale that assigns strings to colors using the pool of colors specified as `values`.


    Parameters
    ----------
    values: :class:`list` of :class:`str`
        The colors to choose when assigning values to colors.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleDiscreteManual("color", values=values)


def scale_fill_discrete():
    """The default discrete fill scale. This maps each discrete value to a fill color.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return scale_fill_hue()


def scale_fill_continuous():
    """The default continuous fill scale. This linearly interpolates colors between the min and max observed values.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorContinuous("fill")


def scale_fill_identity():
    """A color scale that assumes the expression specified in the ``fill`` aesthetic can be used as a fill color.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorContinuousIdentity("fill")


def scale_fill_hue():
    """Map discrete fill colors to evenly placed positions around the color wheel.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.

    """
    return ScaleColorHue("fill")


def scale_fill_manual(*, values):
    """A color scale that assigns strings to fill colors using the pool of colors specified as `values`.


    Parameters
    ----------
    values: :class:`list` of :class:`str`
        The colors to choose when assigning values to colors.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleDiscreteManual("fill", values=values)


def scale_shape_manual(*, values):
    """A scale that assigns shapes to discrete aesthetics. See `the plotly documentation <https://plotly.com/python-api-reference/generated/plotly.graph_objects.scatter.html#plotly.graph_objects.scatter.Marker.symbol>`__ for a list of supported shapes.

    Parameters
    ----------
    values: :class:`list` of :class:`str`
        The shapes from which to choose.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleDiscreteManual("shape", values=values)


def scale_shape_auto():
    """A scale that automatically assigns shapes to discrete aesthetics.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleShapeAuto("shape")

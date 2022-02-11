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

    @abc.abstractmethod
    def is_continuous(self):
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


class ScaleDiscrete(Scale):
    def __init__(self, aesthetic_name):
        super().__init__(aesthetic_name)

    def transform_data(self, field_expr):
        return field_expr

    def is_discrete(self):
        return True

    def is_continuous(self):
        return False


class ScaleColorDiscrete(ScaleDiscrete):
    def transform_data_local(self, df, parent):
        categorical_strings = set(df[self.aesthetic_name])
        unique_color_mapping = categorical_strings_to_colors(categorical_strings, parent)

        new_column_name = f"{self.aesthetic_name}_legend"
        new_df = df.assign(**{new_column_name: df[self.aesthetic_name]})

        new_df[self.aesthetic_name] = new_df[self.aesthetic_name].map(unique_color_mapping)

        return new_df


class ScaleColorContinuous(ScaleContinuous):
    def transform_data_local(self, df, parent):
        color_series = df[self.aesthetic_name]
        color_mapping = continuous_nums_to_colors(color_series, parent.continuous_color_scale)

        df[self.aesthetic_name] = df[self.aesthetic_name].map(lambda i: color_mapping[i])

        return df


# Legend names messed up for scale color identity
class ScaleColorDiscreteIdentity(ScaleDiscrete):
    def transform_data_local(self, data, parent):
        return data


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
    """The default discrete color scale. This maps each discrete value to a color.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorDiscrete("color")


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
    return ScaleColorDiscreteIdentity("color")


def scale_fill_discrete():
    """The default discrete fill scale. This maps each discrete value to a fill color.

    Returns
    -------
    :class:`.FigureAttribute`
        The scale to be applied.
    """
    return ScaleColorDiscrete("fill")


def scale_fill_continuous():
    """The default discrete fill scale. This linearly interpolates colors between the min and max observed values.

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
    return ScaleColorDiscreteIdentity("fill")

from collections.abc import Mapping
import abc

import plotly.graph_objects as go

import hail as hl

class GGPlot:

    def __init__(self, ht, aes, geoms=[]):
        self.ht = ht
        self.aes = aes
        self.geoms = geoms

    # Thinking:
    # Aesthetics are basically just a dictionary of properties, string to either string or hail expression.
    #    The aesthetics on the plot are sort of global aesthetics. Also possible to have geom specfic ones.
    # We have to be able to add geoms to this thing.
    #   A geom has required aesthetics. They can be specified as part of the global aesthetics of the plot object
    #   or be specific to that geom.
    #   Some geoms have implicit aesthetics (geom_histogram makes y a count of the x's, that's basically a group by aggregation)
    #       Worth doing this sort of group by in hail vs pandas/plotly?
    #
    # How does one render a graph?
    # 1. Identify the necessary table fields, and select to only those.
    # 2. Collect those fields (or a subsample of them?) either to a python list or perhaps a pandas dataframe.
    # 3. Generate the plot in plotly based on the assembled information.

    def __add__(self, other):
        assert(isinstance(other, FigureAttribute))

        copied = self.copy()
        if isinstance(other, Geom):
            copied.geoms.append(other)
        else:
            raise ValueError("Not implemented")

        return copied

    def copy(self):
        return GGPlot(self.ht, self.aes, self.geoms[:])

    def render(self):
        # Step 1: Update aesthetics accordingly, all need to point into this table.
        # TODO: Make sure all aesthetics are hail expressions.

        fields_to_select = {"figure_mapping": hl.struct(**self.aes)}
        for geom_idx, geom in enumerate(self.geoms):
            label = f"geom{geom_idx}"
            fields_to_select[label] = hl.struct(**geom.aes)

        selected = self.ht.select(**fields_to_select)
        collected = selected.collect()

        # Ok, so far so good, need to make a plot based on the geoms.
        # Geoms have a notion of scope. They have their own aesthetics, plus the global figure aesthetics.
        # Idea: For each geom's specific fields, create a special struct in the select just for it.

        fig = go.Figure()

        for geom_idx, geom in enumerate(self.geoms):
            geom.apply_to_fig(collected, f"geom{geom_idx}", fig)

        return fig


def ggplot(table, aes):
    return GGPlot(table, aes)


class Aesthetic(Mapping):
    # kwargs values should either be strings or fields of a table. We will have to resolve the fact that all tables
    # need to match the base ggplot table at some point.

    def __init__(self, properties):
        self.properties = properties

    def __getitem__(self, item):
        return self.properties[item]

    def __len__(self):
        return len(self.properties)

    def __iter__(self):
        return iter(self.properties)


def aes(**kwargs):
    return Aesthetic(kwargs)


class FigureAttribute:
    pass


class Geom(FigureAttribute):

    def __init__(self, aes):
        self.aes = aes

    @abc.abstractmethod
    def apply_to_fig(self, parent, mapping_field_name, fig_so_far):
        return


def lookup(struct, mapping_field_name, target):
    maybe = struct["figure_mapping"].get(target, struct[mapping_field_name].get(target))
    if maybe is None:
        raise KeyError(f"aesthetic field {target} expected but not found")
    return maybe

class GeomPoint(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, collected, mapping_field_name, fig_so_far):
        x = [lookup(element, mapping_field_name, "x") for element in collected]
        y = [lookup(element, mapping_field_name, "y") for element in collected]
        fig_so_far.add_scatter(x=x, y=y, mode="markers")


def geom_point(mapping=aes()):
    return GeomPoint(mapping)


class GeomLine(Geom):

    def __init__(self, aes):
        super().__init__(aes)

    def apply_to_fig(self, collected, mapping_field_name, fig_so_far):
        x = [lookup(element, mapping_field_name, "x") for element in collected]
        y = [lookup(element, mapping_field_name, "y") for element in collected]
        fig_so_far.add_scatter(x=x, y=y, mode="lines")


def geom_line(mapping=aes()):
    return GeomLine(mapping)

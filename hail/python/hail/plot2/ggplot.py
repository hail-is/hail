import plotly.graph_objects as go

import hail as hl

from .geoms import Geom, FigureAttribute
from .labels import Labels
from .scale import Scale
from .aes import Aesthetic, aes


class GGPlot:

    def __init__(self, ht, aes, geoms=[], labels=Labels(), x_scale=None, y_scale=None):
        self.ht = ht
        self.aes = aes
        self.geoms = geoms
        self.labels = labels
        self.x_scale = x_scale
        self.y_scale = y_scale

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
        assert(isinstance(other, FigureAttribute) or isinstance(other, Aesthetic))

        copied = self.copy()
        if isinstance(other, Geom):
            copied.geoms.append(other)
        elif isinstance(other, Labels):
            copied.labels = copied.labels.merge(other)
        elif isinstance(other, Scale):
            if other.axis == "x":
                copied.x_scale = other
            elif other.axis == "y":
                copied.y_scale = other
            else:
                raise ValueError("Unrecognized axis in scale")
        elif isinstance(other, Aesthetic):
            copied.aes = Aesthetic({**copied.aes.properties, **other.properties})
        else:
            raise ValueError("Not implemented")

        return copied

    def copy(self):
        return GGPlot(self.ht, self.aes, self.geoms[:], self.labels, self.x_scale, self.y_scale)

    def render(self):
        # Step 1: Update aesthetics accordingly, all need to point into this table.
        # TODO: Make sure all aesthetics are hail expressions.

        fields_to_select = {"figure_mapping": hl.struct(**self.aes)}
        for geom_idx, geom in enumerate(self.geoms):
            label = f"geom{geom_idx}"
            fields_to_select[label] = hl.struct(**geom.aes)

        selected = self.ht.select(**fields_to_select)
        aggregators = {}
        for geom_idx, geom in enumerate(self.geoms):
            label = f"geom{geom_idx}"
            stat = geom.get_stat()

            if "x" in selected[label]:
                x_expr = selected[label]["x"]
            elif "x" in selected["figure_mapping"]:
                x_expr = selected["figure_mapping"]["x"]
            else:
                raise ValueError("There wasn't an x")

            agg = stat.make_agg(x_expr, selected["figure_mapping"], selected[label])
            aggregators[label] = agg


        aggregated = selected.aggregate(hl.struct(**aggregators))

        # Ok, so far so good, need to make a plot based on the geoms.
        # Geoms have a notion of scope. They have their own aesthetics, plus the global figure aesthetics.
        # Idea: For each geom's specific fields, create a special struct in the select just for it.

        fig = go.Figure()

        for geom, agg_result in zip(self.geoms, aggregated.values()):
            geom.apply_to_fig(self, agg_result, fig)

        self.labels.apply_to_fig(fig)
        if self.x_scale is not None:
            self.x_scale.apply_to_fig(fig)
        if self.y_scale is not None:
            self.y_scale.apply_to_fig(fig)

        return fig


def ggplot(table, aes=aes()):
    return GGPlot(table, aes)

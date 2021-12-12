import plotly
import plotly.graph_objects as go

import hail as hl

from .geoms import Geom, FigureAttribute
from .labels import Labels
from .scale import Scale
from .aes import Aesthetic, aes


class GGPlot:

    def __init__(self, ht, aes, geoms=[], labels=Labels(), x_scale=None, y_scale=None,
                 discrete_color_scale=plotly.colors.qualitative.D3, continuous_color_scale=plotly.colors.sequential.Viridis):
        self.ht = ht
        self.aes = aes
        self.geoms = geoms
        self.labels = labels
        # Scales are complicated. When an aesthetic is first added, it creates a scale for itself based on the hail type.
        # Need to separately track whether a scale was added by user or by default.
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.discrete_color_scale = discrete_color_scale
        self.discrete_color_dict = {}
        self.discrete_color_idx = 0
        self.continuous_color_scale = continuous_color_scale

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
        return GGPlot(self.ht, self.aes, self.geoms[:], self.labels, self.x_scale, self.y_scale,
                      self.discrete_color_scale, self.continuous_color_scale)

    def render(self):
        # Step 1: Update aesthetics accordingly, all need to point into this table.
        # TODO: Make sure all aesthetics are hail expressions.

        fields_to_select = {"figure_mapping": hl.struct(**self.aes)}
        for geom_idx, geom in enumerate(self.geoms):
            label = f"geom{geom_idx}"
            fields_to_select[label] = hl.struct(**geom.aes)

        selected = self.ht.select(**fields_to_select)
        aggregators = {}
        labels_to_stats = {}
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
            labels_to_stats[label] = stat

        aggregated = selected.aggregate(hl.struct(**aggregators))

        fig = go.Figure()

        for geom, (label, agg_result) in zip(self.geoms, aggregated.items()):
            listified_agg_result = labels_to_stats[label].listify(agg_result)
            geom.apply_to_fig(self, listified_agg_result, fig)

        # Important to update axes after labels, axes names take precedence.
        self.labels.apply_to_fig(fig)
        if self.x_scale is not None:
            self.x_scale.apply_to_fig(fig)
        if self.y_scale is not None:
            self.y_scale.apply_to_fig(fig)

        return fig


def ggplot(table, aes=aes()):
    return GGPlot(table, aes)

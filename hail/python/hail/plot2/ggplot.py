import plotly
import plotly.graph_objects as go

import hail as hl

from .geoms import Geom, FigureAttribute
from .labels import Labels
from .scale import Scale, scale_x_continuous, scale_x_genomic
from .aes import Aesthetic, aes


class GGPlot:

    def __init__(self, ht, aes, geoms=[], labels=Labels(), scales={},
                 discrete_color_scale=plotly.colors.qualitative.D3, continuous_color_scale=plotly.colors.sequential.Viridis):
        self.ht = ht
        self.aes = aes
        self.geoms = geoms
        self.labels = labels
        self.scales = scales
        self.discrete_color_scale = discrete_color_scale
        self.discrete_color_dict = {}
        self.discrete_color_idx = 0
        self.continuous_color_scale = continuous_color_scale

        self.add_default_scales(aes)

    def __add__(self, other):
        assert(isinstance(other, FigureAttribute) or isinstance(other, Aesthetic))

        copied = self.copy()
        if isinstance(other, Geom):
            copied.geoms.append(other)
        elif isinstance(other, Labels):
            copied.labels = copied.labels.merge(other)
        elif isinstance(other, Scale):
            copied.scales[other.aesthetic_name] = other
        elif isinstance(other, Aesthetic):
            copied.aes = copied.aes.merge(other)
        else:
            raise ValueError("Not implemented")

        return copied

    def add_default_scales(self, aesthetic):
        def is_continuous_type(dtype):
            return dtype in [hl.tint32, hl.tint64, hl.float32, hl.float64]

        def is_genomic_type(dtype):
            return isinstance(dtype, hl.tlocus)

        for aesthetic_str, mapped_expr in aesthetic.items():
            dtype = mapped_expr.dtype
            if aesthetic_str in self.scales:
                pass
            else:
                # We only know how to come up with a few default scales.
                if aesthetic_str == "x":
                    if is_continuous_type(dtype):
                        self.scales["x"] = scale_x_continuous()
                    elif is_genomic_type(dtype):
                        self.scales["x"] = scale_x_genomic()
                    else:
                        # Need to add scale_x_discrete
                        pass
                elif aesthetic_str == "y":
                    if is_continuous_type(dtype):
                        self.scales["y"] = scale_x_continuous()
                    else:
                        # Need to add scale_y_discrete
                        pass

    def copy(self):
        return GGPlot(self.ht, self.aes, self.geoms[:], self.labels, self.scales,
                      self.discrete_color_scale, self.continuous_color_scale)

    def render(self):
        # Step 1: Update aesthetics accordingly, all need to point into this table.
        # TODO: Make sure all aesthetics are hail expressions.

        fields_to_select = {"figure_mapping": hl.struct(**self.aes)}
        for geom_idx, geom in enumerate(self.geoms):
            label = f"geom{geom_idx}"
            fields_to_select[label] = hl.struct(**geom.aes.properties)

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
                import pdb; pdb.set_trace()
                raise ValueError("There wasn't an x")

            if "x" in self.scales:
                x_expr = self.scales["x"].transform_data(x_expr)

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
        if self.scales.get("x") is not None:
            self.scales["x"].apply_to_fig(self, fig)
        if self.scales.get("y") is not None:
            self.scales["y"].apply_to_fig(self, fig)

        return fig


def ggplot(table, aes=aes()):
    return GGPlot(table, aes)

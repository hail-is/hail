import itertools
from pprint import pprint

from plotly.subplots import make_subplots

import hail as hl

from .aes import Aesthetic, aes
from .coord_cartesian import CoordCartesian
from .facets import Faceter
from .geoms import FigureAttribute, Geom
from .labels import Labels
from .scale import (
    Scale,
    ScaleContinuous,
    ScaleDiscrete,
    scale_color_continuous,
    scale_color_discrete,
    scale_fill_continuous,
    scale_fill_discrete,
    scale_shape_auto,
    scale_x_continuous,
    scale_x_discrete,
    scale_x_genomic,
    scale_y_continuous,
    scale_y_discrete,
)
from .utils import check_scale_continuity, is_continuous_type, is_genomic_type


class GGPlot:
    """The class representing a figure created using the ``hail.ggplot`` module.

    Create one by using :func:`.ggplot`.

    .. automethod:: to_plotly
    .. automethod:: show
    .. automethod:: write_image
    """

    def __init__(self, ht, aes, geoms=[], labels=Labels(), coord_cartesian=None, scales=None, facet=None):
        if scales is None:
            scales = {}

        self.ht = ht
        self.aes = aes
        self.geoms = geoms
        self.labels = labels
        self.coord_cartesian = coord_cartesian
        self.scales = scales
        self.facet = facet

        self.add_default_scales(aes)

    def __add__(self, other):
        assert isinstance(other, (Aesthetic, FigureAttribute))

        copied = self.copy()
        if isinstance(other, Geom):
            copied.geoms.append(other)
            copied.add_default_scales(other.aes)
        elif isinstance(other, Labels):
            copied.labels = copied.labels.merge(other)
        elif isinstance(other, CoordCartesian):
            copied.coord_cartesian = other
        elif isinstance(other, Scale):
            copied.scales[other.aesthetic_name] = other
        elif isinstance(other, Aesthetic):
            copied.aes = copied.aes.merge(other)
        elif isinstance(other, Faceter):
            copied.facet = other
        else:
            raise ValueError("Not implemented")

        return copied

    def add_default_scales(self, aesthetic):

        for aesthetic_str, mapped_expr in aesthetic.items():
            dtype = mapped_expr.dtype
            if aesthetic_str not in self.scales:
                is_continuous = is_continuous_type(dtype)
                # We only know how to come up with a few default scales.
                if aesthetic_str == "x":
                    if is_continuous:
                        self.scales["x"] = scale_x_continuous()
                    elif is_genomic_type(dtype):
                        self.scales["x"] = scale_x_genomic(reference_genome=dtype.reference_genome)
                    else:
                        self.scales["x"] = scale_x_discrete()
                elif aesthetic_str == "y":
                    if is_continuous:
                        self.scales["y"] = scale_y_continuous()
                    elif is_genomic_type(dtype):
                        raise ValueError("Don't yet support y axis genomic")
                    else:
                        self.scales["y"] = scale_y_discrete()
                elif aesthetic_str == "color" and not is_continuous:
                    self.scales["color"] = scale_color_discrete()
                elif aesthetic_str == "color" and is_continuous:
                    self.scales["color"] = scale_color_continuous()
                elif aesthetic_str == "fill" and not is_continuous:
                    self.scales["fill"] = scale_fill_discrete()
                elif aesthetic_str == "fill" and is_continuous:
                    self.scales["fill"] = scale_fill_continuous()
                elif aesthetic_str == "shape" and not is_continuous:
                    self.scales["shape"] = scale_shape_auto()
                elif aesthetic_str == "shape" and is_continuous:
                    raise ValueError(
                        "The 'shape' aesthetic does not support continuous "
                        "types. Specify values of a discrete type instead."
                    )
                else:
                    if is_continuous:
                        self.scales[aesthetic_str] = ScaleContinuous(aesthetic_str)
                    else:
                        self.scales[aesthetic_str] = ScaleDiscrete(aesthetic_str)

    def copy(self):
        return GGPlot(self.ht, self.aes, self.geoms[:], self.labels, self.coord_cartesian, self.scales, self.facet)

    def verify_scales(self):
        for aes_key in self.aes.keys():
            check_scale_continuity(self.scales[aes_key], self.aes[aes_key].dtype, aes_key)
        for geom in self.geoms:
            aesthetic_dict = geom.aes.properties
            for aes_key in aesthetic_dict.keys():
                check_scale_continuity(self.scales[aes_key], aesthetic_dict[aes_key].dtype, aes_key)

    def to_plotly(self):
        """Turn the hail plot into a Plotly plot.

        Returns
        -------
        A Plotly figure that can be updated with plotly methods.
        """

        def make_geom_label(geom_idx):
            return f"geom{geom_idx}"

        def select_table():
            fields_to_select = {"figure_mapping": hl.struct(**self.aes)}
            if self.facet is not None:
                fields_to_select["facet"] = self.facet.get_expr_to_group_by()

            for geom_idx, geom in enumerate(self.geoms):
                geom_label = make_geom_label(geom_idx)
                fields_to_select[geom_label] = hl.struct(**geom.aes.properties)

            name, ht = hl.struct(**fields_to_select)._to_table('__fallback')
            return ht.select(**{field: ht[name][field] for field in fields_to_select})

        def collect_mappings_and_precomputed(selected):
            mapping_per_geom = []
            precomputes = {}
            for geom_idx, geom in enumerate(self.geoms):
                geom_label = make_geom_label(geom_idx)

                combined_mapping = selected["figure_mapping"].annotate(**selected[geom_label])

                for key in combined_mapping:
                    if key in self.scales:
                        combined_mapping = combined_mapping.annotate(
                            **{key: self.scales[key].transform_data(combined_mapping[key])}
                        )
                mapping_per_geom.append(combined_mapping)
                precomputes[geom_label] = geom.get_stat().get_precomputes(combined_mapping)

            # Is there anything to precompute?
            should_precompute = any([len(precompute) > 0 for precompute in precomputes.values()])

            if should_precompute:
                precomputed = selected.aggregate(hl.struct(**precomputes))
            else:
                precomputed = hl.Struct(**{key: hl.Struct() for key in precomputes.keys()})

            return mapping_per_geom, precomputed

        def get_aggregation_result(selected, mapping_per_geom, precomputed):
            aggregators = {}
            labels_to_stats = {}
            use_faceting = self.facet is not None
            for geom_idx, combined_mapping in enumerate(mapping_per_geom):
                stat = self.geoms[geom_idx].get_stat()
                geom_label = make_geom_label(geom_idx)
                if use_faceting:
                    agg = hl.agg.group_by(
                        selected.facet, stat.make_agg(combined_mapping, precomputed[geom_label], self.scales)
                    )
                else:
                    agg = stat.make_agg(combined_mapping, precomputed[geom_label], self.scales)
                aggregators[geom_label] = agg
                labels_to_stats[geom_label] = stat

            all_agg_results = selected.aggregate(hl.struct(**aggregators))

            if use_faceting:
                facet_list = list(set(itertools.chain(*[list(x.keys()) for x in all_agg_results.values()])))
                facet_to_idx = {facet: idx for idx, facet in enumerate(facet_list)}
                facet_idx_to_agg_result = {
                    geom_label: {facet_to_idx[facet]: agg_result for facet, agg_result in facet_to_agg_result.items()}
                    for geom_label, facet_to_agg_result in all_agg_results.items()
                }
                num_facets = len(facet_list)
            else:
                facet_idx_to_agg_result = {
                    geom_label: {0: agg_result} for geom_label, agg_result in all_agg_results.items()
                }
                num_facets = 1
                facet_list = None

            return labels_to_stats, facet_idx_to_agg_result, num_facets, facet_list

        self.verify_scales()
        selected = select_table()
        mapping_per_geom, precomputed = collect_mappings_and_precomputed(selected)
        labels_to_stats, aggregated, num_facets, facet_list = get_aggregation_result(
            selected, mapping_per_geom, precomputed
        )

        geoms_and_grouped_dfs_by_facet_idx = []
        for geom, (geom_label, agg_result_by_facet) in zip(self.geoms, aggregated.items()):
            dfs_by_facet_idx = {
                facet_idx: labels_to_stats[geom_label].listify(agg_result)
                for facet_idx, agg_result in agg_result_by_facet.items()
            }
            geoms_and_grouped_dfs_by_facet_idx.append((geom, geom_label, dfs_by_facet_idx))

        # Create scaling functions based on all the data:
        transformers = {}
        for scale in self.scales.values():
            all_dfs = list(
                itertools.chain(
                    *[facet_to_dfs_dict.values() for _, _, facet_to_dfs_dict in geoms_and_grouped_dfs_by_facet_idx]
                )
            )
            transformers[scale.aesthetic_name] = scale.create_local_transformer(all_dfs)

        is_faceted = self.facet is not None
        if is_faceted:
            n_facet_rows, n_facet_cols = self.facet.get_facet_nrows_and_ncols(num_facets)
            subplot_args = {
                "rows": n_facet_rows,
                "cols": n_facet_cols,
                "subplot_titles": [
                    ", ".join([str(fs_value) for fs_value in facet_struct.values()]) for facet_struct in facet_list
                ],
                **self.facet.get_shared_axis_kwargs(),
            }
        else:
            n_facet_rows = 1
            n_facet_cols = 1
            subplot_args = {
                "rows": 1,
                "cols": 1,
            }
        fig = make_subplots(**subplot_args)

        # Need to know what I've added to legend already so we don't do it more than once.
        legend_cache = {}

        for geom, geom_label, facet_to_grouped_dfs in geoms_and_grouped_dfs_by_facet_idx:
            for facet_idx, grouped_dfs in facet_to_grouped_dfs.items():
                scaled_grouped_dfs = []
                for df in grouped_dfs:
                    scales_to_consider = list(df.columns) + list(df.attrs)
                    relevant_aesthetics = [scale_name for scale_name in scales_to_consider if scale_name in self.scales]
                    scaled_df = df
                    for relevant_aesthetic in relevant_aesthetics:
                        scaled_df = transformers[relevant_aesthetic](scaled_df)
                    scaled_grouped_dfs.append(scaled_df)

                facet_row = facet_idx // n_facet_cols + 1
                facet_col = facet_idx % n_facet_cols + 1
                geom.apply_to_fig(
                    scaled_grouped_dfs, fig, precomputed[geom_label], facet_row, facet_col, legend_cache, is_faceted
                )

        # Important to update axes after labels, axes names take precedence.
        self.labels.apply_to_fig(fig)
        if self.scales.get("x") is not None:
            self.scales["x"].apply_to_fig(self, fig)
        if self.scales.get("y") is not None:
            self.scales["y"].apply_to_fig(self, fig)
        if self.coord_cartesian is not None:
            self.coord_cartesian.apply_to_fig(fig)

        fig = fig.update_xaxes(title_font_size=18, ticks="outside")
        fig = fig.update_yaxes(title_font_size=18, ticks="outside")
        fig.update_layout(
            plot_bgcolor="white",
            font_family='Arial, "Open Sans", verdana, sans-serif',
            title_font_size=26,
            xaxis=dict(linecolor="black", showticklabels=True),
            yaxis=dict(linecolor="black", showticklabels=True),
            # axes for plotly subplots are numbered following the pattern [xaxis, xaxis2, xaxis3, ...]
            **{
                f"{var}axis{idx}": {"linecolor": "black", "showticklabels": True}
                for idx in range(2, n_facet_rows + n_facet_cols + 1)
                for var in ["x", "y"]
            },
        )

        return fig

    def show(self):
        """Render and show the plot, either in a browser or notebook."""
        self.to_plotly().show()

    def write_image(self, path):
        """Write out this plot as an image.

        This requires you to have installed the python package kaleido from pypi.

        Parameters
        ----------
        path: :class:`str`
            The path to write the file to.
        """
        self.to_plotly().write_image(path)

    def _repr_html_(self):
        return self.to_plotly()._repr_html_()

    def _debug_print(self):
        print("Ggplot Object:")
        print("Aesthetics")
        pprint(self.aes)
        pprint("Scales:")
        pprint(self.scales)
        print("Geoms:")
        pprint(self.geoms)


def ggplot(table, mapping=aes()):
    """Create the initial plot object.

    This function is the beginning of all plots using the ``hail.ggplot`` interface. Plots are constructed
    by calling this function, then adding attributes to the plot to get the desired result.

    Examples
    --------

    Create a y = x^2 scatter plot

    >>> ht = hl.utils.range_table(10)
    >>> ht = ht.annotate(squared = ht.idx**2)
    >>> my_plot = hl.ggplot.ggplot(ht, hl.ggplot.aes(x=ht.idx, y=ht.squared)) + hl.ggplot.geom_point()

    Parameters
    ----------
    table
        The table containing the data to plot.
    mapping
        Default list of aesthetic mappings from table data to plot attributes.

    Returns
    -------
    :class:`.GGPlot`
    """
    assert isinstance(mapping, Aesthetic)
    return GGPlot(table, mapping)

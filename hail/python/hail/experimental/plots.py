import json
import numpy as np
import pandas as pd

import hail as hl
from bokeh.layouts import gridplot
from bokeh.models import *
from bokeh.palettes import Spectral8
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, transform
from hail.typecheck import *
from hail.utils.hadoop_utils import *


def _collect_scatter_plot_data(
        x: hl.expr.NumericExpression,
        y: hl.expr.NumericExpression,
        fields: Dict[str, hl.expr.Expression] = None,
        n_divisions: int = None,
        missing_label: str =  'NA'
) -> pd.DataFrame:

    expressions = dict()
    if fields is not None:
        expressions.update({k: hl.or_else(v, missing_label) if isinstance(v, hl.expr.StringExpression) else v for k, v in fields.items()})

    if n_divisions is None:
        collect_expr = hl.struct(_x=x, _y=y, **expressions)
        plot_data = [point for point in collect_expr.collect() if point._x is not None and point._y is not None]
        source_pd = pd.DataFrame(plot_data)
    else:
        # FIXME: remove the type conversion logic if/when downsample supports continuous values for labels
        continous_expr = {k: 'int32' for k,v in expressions.items() if isinstance(v, hl.expr.Int32Expression)}
        continous_expr.update({k: 'int64' for k,v in expressions.items() if isinstance(v, hl.expr.Int64Expression)})
        continous_expr.update({k: 'float32' for k, v in expressions.items() if isinstance(v, hl.expr.Float32Expression)})
        continous_expr.update({k: 'float64' for k, v in expressions.items() if isinstance(v, hl.expr.Float64Expression)})
        if continous_expr:
            expressions = {k: hl.str(v) if not isinstance(v, hl.expr.StringExpression) else v for k,v in expressions.items()}
        agg_f = x._aggregation_method()
        res = agg_f(hl.agg.downsample(x, y, label=list(expressions.values()) if expressions else None, n_divisions=n_divisions))
        source_pd = pd.DataFrame([dict(_x=point[0], _y=point[1], **dict(zip(expressions, point[2]))) for point in res])
        source_pd = source_pd.astype(continous_expr, copy=False)

    return source_pd


def _get_categorical_palette(factors: List[str]) -> Dict[str, str]:
    n = max(3, len(factors))
    if n < len(hl.plot.palette):
        palette = hl.plot.palette
    elif n < 21:
        from bokeh.palettes import Category20
        palette = Category20[n]
    else:
        from bokeh.palettes import viridis
        palette = viridis(n)

    return CategoricalColorMapper(factors=factors, palette=palette)


def _get_scatter_plot_elements(sp: Plot, source_pd: pd.DataFrame, label_cols: List[str], colors: Dict[str, ColorMapper] = None):

    if not x.shape[0]:
        print("WARN: No data to plot.")
        return sp, None, None, None, None, None

    sp.tools.append(HoverTool(tooltips=[(x, f'@{x}') for x in source_pd.columns]))

    cds = ColumnDataSource(source_pd)

    if not label_cols:
        sp.circle('_x', '_y', source=cds)
        return sp, None, None, None, None, None

    continuous_cols = [col for col in label_cols if
                       (str(source_pd.dtypes[col]).startswith('float') or
                        str(source_pd.dtypes[col]).startswith('int'))]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    #  Assign color mappers to columns
    if colors is None:
        colors = {}
    color_mappers = {}

    for col in continuous_cols:
        low = np.nanmin(source_pd[col])
        if np.isnan(low):
            low = 0
            high = 0
        else:
            high = np.nanmax(source_pd[col])
        color_mappers[col] = colors[col] if col in colors else LinearColorMapper(palette='Magma256', low=low, high=high)

    for col in factor_cols:
        if col in colors:
            color_mappers[col] = colors[col]
        else:
            factors = list(set(source_pd[col]))
            color_mappers[col] = _get_categorical_palette(factors)

    # Create initial glyphs
    initial_col = label_cols[0]
    initial_mapper = color_mappers[initial_col]
    legend_items = {}

    if not factor_cols:
        all_renderers = [
            sp.circle('_x', '_y', color=transform(initial_col, initial_mapper), source=cds)
        ]

    else:
        all_renderers = []
        legend_items = {col: DefaultDict(list) for col in factor_cols}
        for key in source_pd.groupby(factor_cols).groups.keys():
            key = key if len(factor_cols) > 1 else [key]
            cds_view = CDSView(source=cds, filters=[GroupFilter(column_name=factor_cols[i], group=key[i]) for i in range(0, len(factor_cols))])
            renderer = sp.circle('_x', '_y', color=transform(initial_col, initial_mapper), source=cds, view=cds_view)
            all_renderers.append(renderer)
            for i in range(0, len(factor_cols)):
                legend_items[factor_cols[i]][key[i]].append(renderer)

        legend_items = {factor: [LegendItem(label=key, renderers=renderers) for key, renderers in key_renderers.items()] for factor, key_renderers in legend_items.items()}

    # Add legend / color bar
    legend = Legend(visible=False, click_policy='hide', orientation='vertical') if initial_col not in factor_cols else Legend(items=legend_items[initial_col], click_policy='hide', orientation='vertical')
    color_bar = ColorBar(visible=False) if initial_col not in continuous_cols else ColorBar(color_mapper=color_mappers[initial_col])
    sp.add_layout(legend, 'left')
    sp.add_layout(color_bar, 'left')

    return sp, legend_items, legend, color_bar, color_mappers, all_renderers


def scatter_plot(
        x: hl.expr.NumericExpression,
        y: hl.expr.NumericExpression,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        label_fields: Dict[str, hl.expr.Expression] = None,
        source_fields: Dict[str, hl.expr.Expression] = None,
        colors: Dict[str, ColorMapper] = None,
        width: int = 800,
        height: int = 800,
        n_divisions: int = None,
        missing_label: str = 'NA'
) -> Column:
    """Create an interactive scatter plot.

       ``x`` and ``y`` must both be a :class:`NumericExpression` from the same :class:`Table`.

       This function returns a :class:`bokeh.plotting.figure.Column` containing:
       - a :class:`bokeh.models.widgets.Select` selection widget if multiple entries are specified in the ``label_fields``
       - a :class:`bokeh.plotting.figure.Figure` containing the interactive scatter plot

       Points will be colored by one of the labels defined in the ``label_fields`` using the color scheme defined in
       the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
       mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
       for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
       for continuous labels.
       For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label.
       Note that using many different labelling schemes in the same plots, particularly if those labels contain many
       different classes could slow down the plot interactions.

       Hovering on points will display their coordinates, labels and any additional fields specified in ``source_fields``.

        Parameters
        ----------
        x : :class:`.NumericExpression`
            List of x-values to be plotted.
        y : :class:`.NumericExpression`
            List of y-values to be plotted.
        title : str
            Title of the scatterplot.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        label_fields : Dict[str, :class:`.Expression`]
            Dict of label name -> label value for x and y values.
            Used to color each point.
            When multiple labels are given, a dropdown will be displayed with the different options.
            Can be used with categorical or continuous expressions.
        source_fields : Dict[str, :class:`.Expression`]
            Extra fields to be displayed when hovering over a point on the plot.
        colors : Dict[str, :class:`bokeh.models.mappers.ColorMapper`]
            Dict of label name -> color mapper.
            Used to set colors for the labels defined using ``label_fields``.
            If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
        width: int
            Plot width
        height: int
            Plot height
        n_divisions : int
            Factor by which to downsample. A good starting place is 500; a lower input results in fewer output datapoints.
        missing_label: str
            Label to use when a point is missing data for a categorical label


        Returns
        -------
        :class:`bokeh.plotting.figure.Column`
        """
    source_fields = {} if source_fields is None else source_fields
    label_fields = {} if label_fields is None else label_fields

    label_cols = list(label_fields.keys())

    source_pd = _collect_scatter_plot_data(x, y, fields={**source_fields, **label_fields}, n_divisions=n_divisions, missing_label=missing_label)
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, legend_items, legend, color_bar, color_mappers, scatter_renderers = _get_scatter_plot_elements(sp, source_pd, label_cols, colors)
    plot_elements = [sp]

    if len(label_cols) > 1:
        # JS call back selector
        callback = CustomJS(args=dict(legend_items=legend_items, legend=legend, color_bar=color_bar, color_mappers=color_mappers, scatter_renderers=scatter_renderers), code="""

        for (var i = 0; i < scatter_renderers.length; i++){
            scatter_renderers[i].glyph.fill_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].glyph.line_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
            scatter_renderers[i].visible = true
        }

        if (cb_obj.value in legend_items){
            legend.items=legend_items[cb_obj.value]
            legend.visible=true
            color_bar.visible=false
        }else{
            legend.visible=false
            color_bar.visible=true
        }

        """)

        select = Select(title="Color by", value=label_cols[0], options=label_cols)
        select.js_on_change('value', callback)
        plot_elements.insert(0, select)

    return Column(children=plot_elements)


def joint_plot(
        x: hl.expr.NumericExpression,
        y: hl.expr.NumericExpression,
        title: str = None,
        xlabel: str = None,
        ylabel: str = None,
        label_fields: Dict[str, hl.expr.Expression] = None,
        source_fields: Dict[str, hl.expr.StringExpression] = None,
        colors: Dict[str, ColorMapper] = None,
        width: int = 800,
        height: int = 800,
        n_divisions: int = None,
        missing_label: str = 'NA'
) -> Column:
    """Create an interactive scatter plot with marginal densities on the side.

       ``x`` and ``y`` must both be a :class:`NumericExpression` from the same :class:`Table`.

       This function returns a :class:`bokeh.plotting.figure.Column` containing two :class:`bokeh.plotting.figure.Row`:
       - The first row contains the X-axis marginal density and a selection widget if multiple entries are specified in the ``label_fields``
       - The second row contains the scatter plot and the y-axis marginal density

       Points will be colored by one of the labels defined in the ``label_fields`` using the color scheme defined in
       the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
       mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
       for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
       for continuous labels.
       For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label in the scatter plot.
       Note that using many different labelling schemes in the same plots, particularly if those labels contain many
       different classes could slow down the plot interactions.

       Hovering on points in the scatter plot displays their coordinates, labels and any additional fields specified in ``source_fields``.

        Parameters
        ----------
        x : :class:`.NumericExpression`
            List of x-values to be plotted.
        y : :class:`.NumericExpression`
            List of y-values to be plotted.
        title : str
            Title of the scatterplot.
        xlabel : str
            X-axis label.
        ylabel : str
            Y-axis label.
        label_fields : Dict[str, :class:`.Expression`]
            Dict of label name -> label value for x and y values.
            Used to color each point.
            When multiple labels are given, a dropdown will be displayed with the different options.
            Can be used with categorical or continuous expressions.
        source_fields : Dict[str, :class:`.Expression`]
            Extra fields to be displayed when hovering over a point on the plot.
        colors : Dict[str, :class:`bokeh.models.mappers.ColorMapper`]
            Dict of label name -> color mapper.
            Used to set colors for the labels defined using ``label_fields``.
            If not used at all, or label names not appearing in this dict will be colored using a default color scheme.
        width: int
            Plot width
        height: int
            Plot height
        n_divisions : int
            Factor by which to downsample. A good starting place is 500; a lower input results in fewer output datapoints.
        missing_label: str
            Label to use when a point is missing data for a categorical label


        Returns
        -------
        :class:`bokeh.plotting.figure.Column`
        """
    # Collect data
    source_fields = {} if source_fields is None else source_fields
    label_fields = {} if label_fields is None else label_fields
    label_cols = list(label_fields.keys())
    source_pd = _collect_scatter_plot_data(x, y, fields={**source_fields, **label_fields}, n_divisions=n_divisions, missing_label=missing_label)
    sp = figure(title=title, x_axis_label=xlabel, y_axis_label=ylabel, height=height, width=width)
    sp, legend_items, legend, color_bar, color_mappers, scatter_renderers = _get_scatter_plot_elements(sp, source_pd, label_cols, colors)

    continuous_cols = [col for col in label_cols if
                       (str(source_pd.dtypes[col]).startswith('float') or
                        str(source_pd.dtypes[col]).startswith('int'))]
    factor_cols = [col for col in label_cols if col not in continuous_cols]

    # Density plots
    def get_density_plot_items(
            source_pd,
            p,
            axis,
            colors: Dict[str, ColorMapper],
            continuous_cols: List[str],
            factor_cols: List[str]
    ):
        """
        axis should be either '_x' or '_y'
        """

        density_renderers = []
        max_densities = {}
        if not factor_cols or continuous_cols:
            dens, edges = np.histogram(source_pd[axis], density=True)
            edges = edges[:-1]
            xy = (edges, dens) if axis == '_x' else (dens, edges)
            cds = ColumnDataSource({'x': xy[0], 'y': xy[1]})
            line = p.line('x', 'y', source=cds)
            density_renderers.extend([(col, "", line) for col in continuous_cols])
            max_densities = {col: np.max(dens) for col in continuous_cols}

        for factor_col in factor_cols:
            factor_colors = colors.get(factor_col, _get_categorical_palette(list(set(source_pd[factor_col]))))
            factor_colors = dict(zip(factor_colors.factors, factor_colors.palette))
            density_data = source_pd[[factor_col, axis]].groupby(factor_col).apply(lambda df: np.histogram(df[axis], density=True))
            for factor, (dens, edges) in density_data.iteritems():
                edges = edges[:-1]
                xy = (edges, dens) if axis == '_x' else (dens, edges)
                cds = ColumnDataSource({'x': xy[0], 'y': xy[1]})
                density_renderers.append((factor_col, factor, p.line('x', 'y', color=factor_colors[factor], source=cds)))
                max_densities[factor_col] = np.max(list(dens) + [max_densities.get(factor_col, 0)])

        p.legend.visible = False
        p.grid.visible = False
        p.outline_line_color = None
        return p, density_renderers, max_densities

    xp = figure(title=title, height=int(height / 3), width=width, x_range=sp.x_range)
    xp, x_renderers, x_max_densities = get_density_plot_items(source_pd, xp, axis='_x', colors=color_mappers, continuous_cols=continuous_cols, factor_cols=factor_cols)
    xp.xaxis.visible = False
    yp = figure(height=height, width=int(width / 3), y_range=sp.y_range)
    yp, y_renderers, y_max_densities = get_density_plot_items(source_pd, yp, axis='_y', colors=color_mappers, continuous_cols=continuous_cols, factor_cols=factor_cols)
    yp.yaxis.visible = False
    density_renderers = x_renderers + y_renderers
    first_row = [xp]

    if len(label_cols) > 1:

        for factor_col, factor, renderer in density_renderers:
            renderer.visible = factor_col == label_cols[0]

        if label_cols[0] in factor_cols:
            xp.y_range.start = 0
            xp.y_range.end = x_max_densities[label_cols[0]]
            yp.x_range.start = 0
            yp.x_range.end = y_max_densities[label_cols[0]]

        # JS call back selector
        callback = CustomJS(
            args=dict(
                legend_items=legend_items,
                legend=legend,
                color_bar=color_bar,
                color_mappers=color_mappers,
                scatter_renderers=scatter_renderers,
                density_renderers=x_renderers + y_renderers,
                x_range = xp.y_range,
                x_max_densities=x_max_densities,
                y_range=yp.x_range,
                y_max_densities=y_max_densities
            ), code="""

                for (var i = 0; i < scatter_renderers.length; i++){
                    scatter_renderers[i].glyph.fill_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
                    scatter_renderers[i].glyph.line_color = {field: cb_obj.value, transform: color_mappers[cb_obj.value]}
                    scatter_renderers[i].visible = true
                }
                
                for (var i = 0; i < density_renderers.length; i++){
                    density_renderers[i][2].visible = density_renderers[i][0] == cb_obj.value
                }

                if (cb_obj.value in legend_items){
                    legend.items=legend_items[cb_obj.value]
                    legend.visible=true
                    color_bar.visible=false
                }else{
                    legend.visible=false
                    color_bar.visible=true
                }
                
                x_range.start = 0
                y_range.start = 0
                x_range.end = x_max_densities[cb_obj.value]
                y_range.end = y_max_densities[cb_obj.value]

                """)

        select = Select(title="Color by", value=label_cols[0], options=label_cols)
        select.js_on_change('value', callback)
        first_row.append(select)

    return gridplot(first_row, [sp, yp])



def plot_roc_curve(ht, scores, tp_label='tp', fp_label='fp', colors=None, title='ROC Curve', hover_mode='mouse'):
    """Create ROC curve from Hail Table.

    One or more `score` fields must be provided, which are assessed against `tp_label` and `fp_label` as truth data.

    High scores should correspond to true positives.

    Parameters
    ----------
    ht : :class:`.Table`
        Table with required data
    scores : :obj:`str` or :obj:`list` of :obj:`.str`
        Top-level location of scores in ht against which to generate PR curves.
    tp_label : :obj:`str`
        Top-level location of true positives in ht.
    fp_label : :obj:`str`
        Top-level location of false positives in ht.
    colors : :obj:`dict` of :obj:`str`
        Optional colors to use (score -> desired color).
    title : :obj:`str`
        Title of plot.
    hover_mode : :obj:`str`
        Hover mode; one of 'mouse' (default), 'vline' or 'hline'

    Returns
    -------
    :obj:`tuple` of :class:`.Figure` and :obj:`list` of :obj:`str`
        Figure, and list of AUCs corresponding to scores.
    """
    if colors is None:
        # Get a palette automatically
        from bokeh.palettes import d3
        palette = d3['Category10'][max(3, len(scores))]
        colors = {score: palette[i] for i, score in enumerate(scores)}

    if isinstance(scores, str):
        scores = [scores]
    total_tp, total_fp = ht.aggregate((hl.agg.count_where(ht[tp_label]), hl.agg.count_where(ht[fp_label])))

    p = figure(title=title, x_axis_label='FPR', y_axis_label='TPR', tools="hover,save,pan,box_zoom,reset,wheel_zoom")
    p.add_layout(Title(text=f'Based on {total_tp} TPs and {total_fp} FPs'), 'above')

    aucs = []
    for score in scores:
        ordered_ht = ht.key_by(_score=-ht[score])
        ordered_ht = ordered_ht.select(
            score_name=score, score=ordered_ht[score],
            tpr=hl.scan.count_where(ordered_ht[tp_label]) / total_tp,
            fpr=hl.scan.count_where(ordered_ht[fp_label]) / total_fp,
        ).key_by().drop('_score')
        last_row = hl.utils.range_table(1).key_by().select(score_name=score, score=hl.float64(float('-inf')), tpr=hl.float32(1.0), fpr=hl.float32(1.0))
        ordered_ht = ordered_ht.union(last_row)
        ordered_ht = ordered_ht.annotate(
            auc_contrib=hl.or_else((ordered_ht.fpr - hl.scan.max(ordered_ht.fpr)) * ordered_ht.tpr, 0.0)
        )
        auc = ordered_ht.aggregate(hl.agg.sum(ordered_ht.auc_contrib))
        aucs.append(auc)
        df = ordered_ht.annotate(score_name=ordered_ht.score_name + f' (AUC = {auc:.4f})').to_pandas()
        p.line(x='fpr', y='tpr', legend='score_name', source=ColumnDataSource(df), color=colors[score], line_width=3)

    p.legend.location = 'bottom_right'
    p.legend.click_policy = 'hide'
    p.select_one(HoverTool).tooltips = [(x, f"@{x}") for x in ('score_name', 'score', 'tpr', 'fpr')]
    p.select_one(HoverTool).mode = hover_mode
    return p, aucs


@typecheck(t_path=str)
def hail_metadata(t_path):
    """Create a metadata plot for a Hail Table or MatrixTable.

    Parameters
    ----------
    t_path : str
        Path to the Hail Table or MatrixTable files.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure` or :class:`bokeh.models.widgets.panels.Tabs` or :class:`bokeh.models.layouts.Column`
    """
    def get_rows_data(rows_files):
        file_sizes = []
        partition_bounds = []
        parts_file = [x['path'] for x in rows_files if x['path'].endswith('parts')]
        if parts_file:
            parts = hadoop_ls(parts_file[0])
            for i, x in enumerate(parts):
                index = x['path'].split(f'{parts_file[0]}/part-')[1].split('-')[0]
                if i < len(parts) - 1:
                    test_index = parts[i + 1]['path'].split(f'{parts_file[0]}/part-')[1].split('-')[0]
                    if test_index == index:
                        continue
                file_sizes.append(x['size_bytes'])
        metadata_file = [x['path'] for x in rows_files if x['path'].endswith('metadata.json.gz')]
        if metadata_file:
            with hadoop_open(metadata_file[0], 'rb') as f:
                rows_meta = json.loads(f.read())
                try:
                    partition_bounds = [
                        (x['start']['locus']['contig'], x['start']['locus']['position'],
                         x['end']['locus']['contig'], x['end']['locus']['position'])
                        for x in rows_meta['jRangeBounds']]
                except KeyError:
                    pass
        return partition_bounds, file_sizes

    def scale_file_sizes(file_sizes):
        min_file_size = min(file_sizes) * 1.1
        total_file_size = sum(file_sizes)
        all_scales = [
            ('T', 1e12),
            ('G', 1e9),
            ('M', 1e6),
            ('K', 1e3),
            ('', 1e0)
        ]
        for overall_scale, overall_factor in all_scales:
            if total_file_size > overall_factor:
                total_file_size /= overall_factor
                break
        for scale, factor in all_scales:
            if min_file_size > factor:
                file_sizes = [x / factor for x in file_sizes]
                break
        total_file_size = f'{total_file_size:.1f} {overall_scale}B'
        return total_file_size, file_sizes, scale

    files = hadoop_ls(t_path)

    rows_file = [x['path'] for x in files if x['path'].endswith('rows')]
    entries_file = [x['path'] for x in files if x['path'].endswith('entries')]
    success_file = [x['modification_time'] for x in files if x['path'].endswith('SUCCESS')]

    metadata_file = [x['path'] for x in files if x['path'].endswith('metadata.json.gz')]
    if not metadata_file:
        raise FileNotFoundError('No metadata.json.gz file found.')

    with hadoop_open(metadata_file[0], 'rb') as f:
        overall_meta = json.loads(f.read())
        rows_per_partition = overall_meta['components']['partition_counts']['counts']

    if not rows_file:
        raise FileNotFoundError('No rows directory found.')
    rows_files = hadoop_ls(rows_file[0])

    data_type = 'Table'
    if entries_file:
        data_type = 'MatrixTable'
        rows_file = [x['path'] for x in rows_files if x['path'].endswith('rows')]
        rows_files = hadoop_ls(rows_file[0])
    row_partition_bounds, row_file_sizes = get_rows_data(rows_files)

    total_file_size, row_file_sizes, row_scale = scale_file_sizes(row_file_sizes)

    panel_size = 480
    subpanel_size = 120

    if not row_partition_bounds:
        warnings.warn('Table is not partitioned. Only plotting file sizes')
        row_file_sizes_hist, row_file_sizes_edges = np.histogram(row_file_sizes, bins=50)
        p_file_size = figure(plot_width=panel_size, plot_height=panel_size)
        p_file_size.quad(right=row_file_sizes_hist, left=0, bottom=row_file_sizes_edges[:-1],
                         top=row_file_sizes_edges[1:], fill_color="#036564", line_color="#033649")
        p_file_size.yaxis.axis_label = f'File size ({row_scale}B)'
        return p_file_size

    all_data = {
        'partition_widths': [-1 if x[0] != x[2] else x[3] - x[1] for x in row_partition_bounds],
        'partition_bounds': [f'{x[0]}:{x[1]}-{x[2]}:{x[3]}' for x in row_partition_bounds],
        'spans_chromosome': ['Spans chromosomes' if x[0] != x[2] else 'Within chromosome' for x in row_partition_bounds],
        'row_file_sizes': row_file_sizes,
        'row_file_sizes_human': [f'{x:.1f} {row_scale}B' for x in row_file_sizes],
        'rows_per_partition': rows_per_partition,
        'index': list(range(len(rows_per_partition)))
    }

    if entries_file:
        entries_rows_files = hadoop_ls(entries_file[0])
        entries_rows_file = [x['path'] for x in entries_rows_files if x['path'].endswith('rows')]
        if entries_rows_file:
            entries_files = hadoop_ls(entries_rows_file[0])
            entry_partition_bounds, entry_file_sizes = get_rows_data(entries_files)
            total_entry_file_size, entry_file_sizes, entry_scale = scale_file_sizes(entry_file_sizes)
            all_data['entry_file_sizes'] = entry_file_sizes
            all_data['entry_file_sizes_human'] = [f'{x:.1f} {entry_scale}B' for x in row_file_sizes]

    title = f'{data_type}: {t_path}'

    msg = f"Rows: {sum(all_data['rows_per_partition']):,}<br/>Partitions: {len(all_data['rows_per_partition']):,}<br/>Size: {total_file_size}<br/>"
    if success_file[0]:
        msg += success_file[0]

    tools = "hover,save,pan,box_zoom,reset,wheel_zoom"

    source = ColumnDataSource(pd.DataFrame(all_data))
    p = figure(tools=tools, plot_width=panel_size, plot_height=panel_size)
    p.title.text = title
    p.xaxis.axis_label = 'Number of rows'
    p.yaxis.axis_label = f'File size ({row_scale}B)'
    color_map = factor_cmap('spans_chromosome', palette=Spectral8,
                            factors=list(set(all_data['spans_chromosome'])))
    p.scatter('rows_per_partition', 'row_file_sizes', color=color_map, legend='spans_chromosome', source=source)
    p.legend.location = 'bottom_right'
    p.select_one(HoverTool).tooltips = [(x, f'@{x}') for x in
                                        ('rows_per_partition', 'row_file_sizes_human', 'partition_bounds', 'index')]

    p_stats = Div(text=msg)
    p_rows_per_partition = figure(x_range=p.x_range, plot_width=panel_size, plot_height=subpanel_size)
    p_file_size = figure(y_range=p.y_range, plot_width=subpanel_size, plot_height=panel_size)

    rows_per_partition_hist, rows_per_partition_edges = np.histogram(all_data['rows_per_partition'], bins=50)
    p_rows_per_partition.quad(top=rows_per_partition_hist, bottom=0, left=rows_per_partition_edges[:-1],
                              right=rows_per_partition_edges[1:],
                              fill_color="#036564", line_color="#033649")
    row_file_sizes_hist, row_file_sizes_edges = np.histogram(all_data['row_file_sizes'], bins=50)
    p_file_size.quad(right=row_file_sizes_hist, left=0, bottom=row_file_sizes_edges[:-1],
                     top=row_file_sizes_edges[1:], fill_color="#036564", line_color="#033649")

    rows_grid = gridplot([[p_rows_per_partition, p_stats], [p, p_file_size]])

    if 'entry_file_sizes' in all_data:
        title = f'Statistics for {data_type}: {t_path}'

        msg = f"Rows: {sum(all_data['rows_per_partition']):,}<br/>Partitions: {len(all_data['rows_per_partition']):,}<br/>Size: {total_entry_file_size}<br/>"
        if success_file[0]:
            msg += success_file[0]

        source = ColumnDataSource(pd.DataFrame(all_data))
        p = figure(tools=tools, plot_width=panel_size, plot_height=panel_size)
        p.title.text = title
        p.xaxis.axis_label = 'Number of rows'
        p.yaxis.axis_label = f'File size ({entry_scale}B)'
        color_map = factor_cmap('spans_chromosome', palette=Spectral8, factors=list(set(all_data['spans_chromosome'])))
        p.scatter('rows_per_partition', 'entry_file_sizes', color=color_map, legend='spans_chromosome', source=source)
        p.legend.location = 'bottom_right'
        p.select_one(HoverTool).tooltips = [(x, f'@{x}') for x in ('rows_per_partition', 'entry_file_sizes_human', 'partition_bounds', 'index')]

        p_stats = Div(text=msg)
        p_rows_per_partition = figure(x_range=p.x_range, plot_width=panel_size, plot_height=subpanel_size)
        p_rows_per_partition.quad(top=rows_per_partition_hist, bottom=0, left=rows_per_partition_edges[:-1],
                                  right=rows_per_partition_edges[1:],
                                  fill_color="#036564", line_color="#033649")
        p_file_size = figure(y_range=p.y_range, plot_width=subpanel_size, plot_height=panel_size)

        row_file_sizes_hist, row_file_sizes_edges = np.histogram(all_data['entry_file_sizes'], bins=50)
        p_file_size.quad(right=row_file_sizes_hist, left=0, bottom=row_file_sizes_edges[:-1],
                         top=row_file_sizes_edges[1:], fill_color="#036564", line_color="#033649")
        entries_grid = gridplot([[p_rows_per_partition, p_stats], [p, p_file_size]])

        return Tabs(tabs=[Panel(child=entries_grid, title='Entries'), Panel(child=rows_grid, title='Rows')])
    else:
        return rows_grid

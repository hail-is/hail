import json
import numpy as np
import pandas as pd

import hail as hl
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import *
from bokeh.palettes import Spectral8
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, HoverTool
from hail.typecheck import *
from hail.utils.hadoop_utils import *



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

def genotype_array_cluster_plot(mt, positional_input, x_type='NORMX', y_type='NORMY', gt_type='GT', show_plot=False, show_ellipse=True):
    """Create a (set of) scatterplot(s) showing SNP clusters per variant specified.

    Parameters
    ----------
    mt : hail matrix table with locus, gt_type, x_type, y_typee, THETA values
    positional_input : either a type IntervalExpression or list of positions (can be 1)"
    show_plot : default False, advised to only use this for small number of variants.

    Dependencies
    -----------
    Script requires : hail, bokeh.plotting, bokeh.io (), bokeh.models (ColumnDataSource, HoverTool), bokeh.models.tools

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure` OR a list of :class:`bokeh.plotting.figure.Figure`'s

    """

    def cluster_plot(position, x_type, y_type, show_plot):
        mt_variant = mt.filter_rows(mt['locus'] == hl.eval(position))
        mt_variant_df = mt_variant.entries().to_pandas()

        #Partition each genotype call
        GT_HET = mt_variant_df.loc[(mt_variant_df[gt_type] == '1/0') | (mt_variant_df[gt_type] == '0/1')]
        GT_HOM_DOM = mt_variant_df.loc[mt_variant_df[gt_type] == '0/0']
        GT_HOM_R = mt_variant_df.loc[mt_variant_df[gt_type] == '1/1']
        GT_NA = mt_variant_df.loc[mt_variant_df[gt_type] == './.']

        #For ellipse : generate mean angle and width for each GT partition
        HET_THETA = GT_HET.THETA.mean()
        HOM_DOM_THETA = GT_HOM_DOM.THETA.mean()
        HOM_R_THETA = GT_HOM_R.THETA.mean()
        HET_MEAN_X = GT_HET[x_type].mean()
        HET_MEAN_Y = GT_HET[y_type].mean()
        HOM_DOM_MEAN_X = GT_HOM_DOM[x_type].mean()
        HOM_DOM_MEAN_Y = GT_HOM_DOM[y_type].mean()
        HOM_R_MEAN_X = GT_HOM_R[x_type].mean()
        HOM_R_MEAN_Y = GT_HOM_R[y_type].mean()

        def max_distance(values):
            v = max(values)-min(values)
            return v

        #For ellipse : Take [1% to 99% Quantile] of values, this minimizes noise
        HET_QUANT_X = GT_HET[x_type].quantile([.01, .99])
        HET_QUANT_Y = GT_HET[y_type].quantile([.01, .99])
        HOM_DOM_QUANT_X = GT_HOM_DOM[x_type].quantile([.01, .99])
        HOM_DOM_QUANT_Y = GT_HOM_DOM[y_type].quantile([.01, .99])
        HOM_R_QUANT_X = GT_HOM_R[x_type].quantile([.01, .99])
        HOM_R_QUANT_Y = GT_HOM_R[y_type].quantile([.01, .99])

        #For ellipse : take maximum distance between the genotype calls once noise is removed
        #this gives width and height for ellipse
        HET_MAX_X = max_distance(HET_QUANT_X)
        HET_MAX_Y = max_distance(HET_QUANT_Y)
        HOM_DOM_MAX_X = max_distance(HOM_DOM_QUANT_X)
        HOM_DOM_MAX_Y = max_distance(HOM_DOM_QUANT_Y)
        HOM_R_MAX_X = max_distance(HOM_R_QUANT_X)
        HOM_R_MAX_Y = max_distance(HOM_R_QUANT_Y)

        df = pd.DataFrame(
            {
                "GT_call": mt_variant_df[gt_type],
                "HET_X": GT_HET[x_type],
                "HET_Y": GT_HET[y_type],
                "HOM_DOM_X": GT_HOM_DOM[x_type],
                "HOM_DOM_Y": GT_HOM_DOM[y_type],
                "HOM_R_X": GT_HOM_R[x_type],
                "HOM_R_Y": GT_HOM_R[y_type],
                "GT_NA_Y": GT_NA[x_type],
                "GT_NA_X": GT_NA[y_type],
                "HET_X" : HET_MEAN_X,
                "HET_Y": HET_MEAN_Y,
                "HET_THETA" : HET_THETA,
                "HET_W" : HET_MAX_X,
                "HET_H" : HET_MAX_Y,
                "HOM_DOM_X" : HOM_DOM_MEAN_X,
                "HOM_DOM_Y": HOM_DOM_MEAN_Y,
                "HOM_DOM_THETA" : HOM_DOM_THETA,
                "HOM_DOM_W" : HOM_DOM_MAX_X,
                "HOM_DOM_H" : HOM_DOM_MAX_Y,
                "HOM_R_X" : HOM_R_MEAN_X,
                "HOM_R_Y": HOM_R_MEAN_Y,
                "HOM_R_THETA" : HOM_R_THETA,
                "HOM_R_W" : HOM_R_MAX_X,
                "HOM_R_H" : HOM_R_MAX_Y
            }
        )
        df
        df = df.fillna('')

        source = ColumnDataSource.from_df(df)
        hover = HoverTool(
            tooltips=[
                ('Sample ID', '@Sample_ID'),
                ('GT Call', '@GT_call'),

            ]
        )
        p = figure(tools=[hover], plot_width=1000, plot_height=1000, title= str(position) + ' Varient Genotype Cluster Plot')
        p.xaxis.axis_label= x_type
        p.yaxis.axis_label= y_type
        p.scatter(
            'HET_X', 'HET_Y', source=source, fill_color='red', line_color='white')
        p.scatter(
            'HOM_DOM_X', 'HOM_DOM_Y', source=source, fill_color='navy', line_color='white')
        p.scatter(
            'HOM_R_X', 'HOM_R_Y', source=source, fill_color='green', line_color='white')
        p.scatter(
            'GT_NA_X', 'GT_NA_Y', source=source, fill_color='black', line_color='white')

        HET_glyph = Ellipse(x=HET_MEAN_X, y=HET_MEAN_Y, width=HET_MAX_Y, height=HET_MAX_X, angle=(HET_THETA*(180/pi)), fill_color="#cab2d6", fill_alpha = 0.05)
        p.add_glyph(HET_glyph)

        HOM_DOM_glyph = Ellipse(x=HOM_DOM_MEAN_X, y=HOM_DOM_MEAN_Y, width=HOM_DOM_MAX_Y, height=HOM_DOM_MAX_X, angle=-(HOM_DOM_THETA*(180/pi)), fill_color="#cab2d6", fill_alpha = 0.05)
        p.add_glyph(HOM_DOM_glyph)

        HOM_R_glyph = Ellipse(x=HOM_R_MEAN_X, y=HOM_R_MEAN_Y, width=HOM_R_MAX_Y, height=HOM_R_MAX_X, angle=-(HOM_R_THETA*(180/pi)), fill_color="#cab2d6", fill_alpha = 0.05)
        p.add_glyph(HOM_R_glyph)

        p.legend.glyph_width = 50
        p.legend.border_line_width = 0.5
        p.legend.location = "top_center"
        p.legend.orientation = "horizontal"
        p.legend.background_fill_alpha = 0.2
        p.legend.background_fill_color = 'grey'

        if show_plot is True:
            show(p)
        return p

    list_p = []
    if type(positional_input) is list:
        for value in positional_input:
            if value.dtype == hl.tinterval:
                if not isinstance(value.dtype._point_type, hl.tlocus):
                    raise ValueError("Wrong point type for interval input. Expecting locus as point type.")
                filtered_mt = hl.filter_intervals(mt, [hl.parse_locus_interval(x) for x in intervals])
                variant_listing = filtered_mt.locus.collect()
                for variant in variant_listing:
                    variant_string = variant_listing[variant].contig + ':' + variant_listing[variant].position
                    new_plot = cluster_plot(variant_string, x_type, y_type, show_plot)
                    list_p.append(new_plot)
            elif value.dtype == hl.tlocus:
                new_plot = cluster_plot(x, t)
                list_p.append(new_plot)
        return list_p
    elif isinstance(positional_input.dtype, hl.tinterval):
        if not isinstance(positional_input.dtype._point_type, hl.tlocus):
            raise ValueError("Wrong point type for interval input. Expecting locus as point type.")
        filtered_mt = hl.filter_intervals(mt, hl.parse_locus_interval(hl.eval(positional_input)))
        variant_listing = filtered_mt.locus.collect()
        for variant in variant_listing:
            variant_string = variant_listing[variant].contig + ':' + variant_listing[variant].position
            new_plot = cluster_plot(variant_string, x_type, y_type, show_plot)
            list_p.append(new_plot)
        return list_p
    elif isinstance(positional_input.dtype, hl.tlocus):
        new_plot = cluster_plot(positional_input, x_type, y_type, show_plot)
        return new_plot
    else:
        raise ValueError("Not a valid variant type. Expecting tinterval, tlocus or Array of the two.")

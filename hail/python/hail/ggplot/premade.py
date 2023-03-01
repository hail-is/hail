import math
from typing import Tuple, Dict, Union

import plotly.express as px
import plotly.graph_objects as go

import hail
from hail import Table
from hail.expr.expressions import expr_float64, expr_locus, expr_any, expr_numeric, Expression, NumericExpression
from hail.plot.plots import _collect_scatter_plot_data
from hail.typecheck import nullable, typecheck, numeric, dictof, oneof, sized_tupleof


@typecheck(pvals=expr_float64, locus=nullable(expr_locus()), title=nullable(str),
           size=int, hover_fields=nullable(dictof(str, expr_any)), collect_all=bool, n_divisions=int,
           significance_line=nullable(numeric))
def manhattan_plot(pvals, locus=None, title=None, size=5, hover_fields=None, collect_all=False, n_divisions=500,
                   significance_line=5e-8):
    """Create a Manhattan plot. (https://en.wikipedia.org/wiki/Manhattan_plot)

    Parameters
    ----------
    pvals : :class:`.Float64Expression`
        P-values to be plotted.
    locus : :class:`.LocusExpression`
        Locus values to be plotted.
    title : str
        Title of the plot.
    size : int
        Size of markers in screen space units.
    hover_fields : Dict[str, :class:`.Expression`]
        Dictionary of field names and values to be shown in the HoverTool of the plot.
    collect_all : bool
        Whether to collect all values or downsample before plotting.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
    significance_line : float, optional
        p-value at which to add a horizontal, dotted red line indicating
        genome-wide significance.  If ``None``, no line is added.

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure`
    """
    if locus is None:
        locus = pvals._indices.source.locus

    ref = locus.dtype.reference_genome

    if hover_fields is None:
        hover_fields = {}

    hover_fields['locus'] = hail.str(locus)
    hover_fields['contig_even'] = locus.contig_idx % 2

    pvals = -hail.log10(pvals)

    source_pd = _collect_scatter_plot_data(
        ('_global_locus', locus.global_position()),
        ('_pval', pvals),
        fields=hover_fields,
        n_divisions=None if collect_all else n_divisions
    )
    source_pd['p_value'] = [10 ** (-p) for p in source_pd['_pval']]
    source_pd['_contig'] = [locus.split(":")[0] for locus in source_pd['locus']]

    observed_contigs = set(source_pd['_contig'])
    observed_contigs = [contig for contig in ref.contigs.copy() if contig in observed_contigs]

    contig_ticks = [ref._contig_global_position(contig) + ref.contig_length(contig) // 2 for contig in observed_contigs]

    extra_hover = {k: True for k in hover_fields if k not in ('locus', 'contig_even')}
    fig = px.scatter(source_pd, x='_global_locus', y='_pval', color='contig_even',
                     labels={
                         '_global_locus': 'global position',
                         '_pval': '-log10 p-value',
                         'p_value': 'p-value',
                         'locus': 'locus'
                     },
                     color_continuous_scale=["#00539C", "#EEA47F"],
                     template='ggplot2',
                     hover_name="locus",
                     hover_data={'contig_even': False,
                                 '_global_locus': False,
                                 '_pval': False,
                                 'p_value': ':.2e',
                                 **extra_hover})

    fig.update_layout(
        xaxis_title="Genomic Coordinate",
        yaxis_title="-log10 p-value",
        title=title,
        showlegend=False,
        coloraxis_showscale=False,
        xaxis=dict(
            tickmode='array',
            tickvals=contig_ticks,
            ticktext=observed_contigs
        )
    )

    fig.update_traces(marker=dict(size=size),
                      selector=dict(mode='markers'))

    if significance_line is not None:
        fig.add_hline(y=-math.log10(significance_line), line_dash='dash', line_color='red', opacity=1, line_width=2)

    return fig


@typecheck(pvals=oneof(expr_numeric, sized_tupleof(str, expr_numeric)),
           label=nullable(oneof(dictof(str, expr_any), expr_any)), title=nullable(str),
           xlabel=nullable(str), ylabel=nullable(str), size=int,
           hover_fields=nullable(dictof(str, expr_any)),
           width=int, height=int, collect_all=bool, n_divisions=nullable(int), missing_label=str)
def qq_plot(
        pvals: Union[NumericExpression, Tuple[str, NumericExpression]],
        label: Union[Expression, Dict[str, Expression]] = None,
        title: str = 'Q-Q plot',
        xlabel: str = 'Expected -log10(p)',
        ylabel: str = 'Observed -log10(p)',
        size: int = 6,
        hover_fields: Dict[str, Expression] = None,
        width: int = 800,
        height: int = 800,
        collect_all: bool = False,
        n_divisions: int = 500,
        missing_label: str = 'NA'
):
    """Create a Quantile-Quantile plot. (https://en.wikipedia.org/wiki/Q-Q_plot)

    ``pvals`` must be either:
    - a :class:`.NumericExpression`
    - a tuple (str, :class:`.NumericExpression`). If passed as a tuple the first element is used as the hover label.

    If no label or a single label is provided, then returns :class:`bokeh.plotting.figure.Figure`
    Otherwise returns a :class:`bokeh.models.layouts.Column` containing:
    - a :class:`bokeh.models.widgets.inputs.Select` dropdown selection widget for labels
    - a :class:`bokeh.plotting.figure.Figure` containing the interactive qq plot

    Points will be colored by one of the labels defined in the ``label`` using the color scheme defined in
    the corresponding entry of ``colors`` if provided (otherwise a default scheme is used). To specify your color
    mapper, check `the bokeh documentation <https://bokeh.pydata.org/en/latest/docs/reference/colors.html>`__
    for CategoricalMapper for categorical labels, and for LinearColorMapper and LogColorMapper
    for continuous labels.
    For categorical labels, clicking on one of the items in the legend will hide/show all points with the corresponding label.
    Note that using many different labelling schemes in the same plots, particularly if those labels contain many
    different classes could slow down the plot interactions.

    Hovering on points will display their coordinates, labels and any additional fields specified in ``hover_fields``.

    Parameters
    ----------
    pvals : :class:`.NumericExpression` or (str, :class:`.NumericExpression`)
        List of x-values to be plotted.
    label : :class:`.Expression` or Dict[str, :class:`.Expression`]]
        Either a single expression (if a single label is desired), or a
        dictionary of label name -> label value for x and y values.
        Used to color each point w.r.t its label.
        When multiple labels are given, a dropdown will be displayed with the different options.
        Can be used with categorical or continuous expressions.
    title : str
        Title of the scatterplot.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    size : int
        Size of markers in screen space units.
    hover_fields : Dict[str, :class:`.Expression`]
        Extra fields to be displayed when hovering over a point on the plot.
    width: int
        Plot width
    height: int
        Plot height
    collect_all : bool
        Whether to collect all values or downsample before plotting.
    n_divisions : int
        Factor by which to downsample (default value = 500). A lower input results in fewer output datapoints.
    missing_label: str
        Label to use when a point is missing data for a categorical label

    Returns
    -------
    :class:`bokeh.plotting.figure.Figure` if no label or a single label was given, otherwise :class:`bokeh.models.layouts.Column`
    """
    hover_fields = {} if hover_fields is None else hover_fields
    label = {} if label is None else {'label': label} if isinstance(label, Expression) else label
    source = pvals._indices.source
    if 'locus' in source.row:
        hover_fields['__locus'] = source['locus']

    if isinstance(source, Table):
        ht = source.select(p_value=pvals, **hover_fields, **label)
    else:
        ht = source.select_rows(p_value=pvals, **hover_fields, **label).rows()
    ht = ht.key_by().select('p_value', *hover_fields, *label).key_by('p_value').persist()
    n = ht.count()
    ht = ht.annotate(
        observed_p=-hail.log10(ht['p_value']),
        expected_p=-hail.log10((hail.scan.count() + 1) / n)
    ).persist()

    if 'p_value' not in hover_fields:
        hover_fields['p_value'] = ht.p_value

    df = _collect_scatter_plot_data(
        ('expected p-value', ht.expected_p),
        ('observed p-value', ht.observed_p),
        fields={k: ht[k] for k in hover_fields},
        n_divisions=None if collect_all else n_divisions,
        missing_label=missing_label
    )

    fig = px.scatter(df, x='expected p-value', y='observed p-value',
                     template='ggplot2',
                     hover_name="__locus",
                     hover_data={**{k: True for k in hover_fields},
                                 '__locus': False,
                                 'p_value': ':.2e'})

    fig.update_traces(marker=dict(size=size, color='black'),
                      selector=dict(mode='markers'))

    from hail.methods.statgen import _lambda_gc_agg
    lambda_gc, max_p = ht.aggregate(
        (_lambda_gc_agg(ht['p_value']), hail.agg.max(hail.max(ht.observed_p, ht.expected_p))))
    fig.add_trace(go.Scatter(x=[0, max_p + 1], y=[0, max_p + 1],
                             mode='lines',
                             name='expected',
                             line=dict(color='red', width=3, dash='dot')))

    label_color = 'red' if lambda_gc > 1.25 else 'orange' if lambda_gc > 1.1 else 'black'

    lgc_label = f'<b>λ GC: {lambda_gc:.2f}</b>'

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        xaxis_title=xlabel,
        xaxis_range=[0, max_p + 1],
        yaxis_title=ylabel,
        yaxis_range=[0, max_p + 1],
        title=title,
        showlegend=False,
        annotations=[
            go.layout.Annotation(
                font=dict(color=label_color, size=22),
                text=lgc_label,
                xanchor='left',
                yanchor='bottom',
                showarrow=False,
                x=max_p * 0.8,
                y=1
            )
        ],
        hovermode="x unified"
    )

    return fig

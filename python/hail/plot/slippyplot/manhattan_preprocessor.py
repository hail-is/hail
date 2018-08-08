import hail as hl
from hail.expr.expressions.expression_utils import check_col_indexed
from hail.expr.expressions import *

def format_manhattan(locus_expr, phenotype_expr, pval_expr, colors=None,
                     threshold=.001):
    """
    Annotate the matrix table with data needed for manhattan plotting.

    * global locus positions (x-axis values)
    * negative log of p_values (y-axis values)
    * colors (because matplotlib will not take a color map)
    * min and max global position (x-axis range)
    * min and max -log(p_value) for each phenotype (y-axis ranges)

    :param locus_expr: :class:`.LocusExpression`
        Row-indexed locus expression on matrix table.
    :param phenotype_expr: :class:`.StringExpression`
        Column-indexed phenotype expression on matrix table.
    :param pval_expr: :class:`.Float32Expression`
        Entry-indexed p-value expression on matrix table.
    :param colors:  :class:`dict` of contigs to hex colors
    :param threshold:
        p-value threshold for hover data (must be kept small so
        that json files don't get too large)
    :return: :class:`.MatrixTable` with manhattan data
    """
    check_row_indexed('format_manhattan', locus_expr)
    check_col_indexed('format_manhattan', phenotype_expr)
    check_entry_indexed('format_manhattan', pval_expr)


    mt = matrix_table_source('format_manhattan', locus_expr)

    row_fields = list(mt.row)
    if 'gene' not in row_fields:
        raise ValueError("expected matrix table to have a `gene` field")

    locus_field = mt._fields_inverse[locus_expr]
    phenotype_field = mt._fields_inverse[phenotype_expr]
    pval_field = mt._fields_inverse[pval_expr]

    # colors = {
    #    '1': "#08ad4d", '2': "#cc0648", '3': "#bbdd11", '4': "#4a87d6",
    #    '5': "#6f50b7", '6': "#e0c10f", '7': "#d10456", '8': "#2779d8",
    #    '9': "#9e0631", '10': "#5fcc06", '11': "#4915a8", '12': "#0453d3",
    #    '13': "#7faf26", '14': "#d17b0c", '15': "#526d13", '16': "#e82019",
    #    '17': "#125b07", '18': "#12e2c3", '19': "#914ae2", '20': "#95ce10",
    #    '21': "#af1ca8", '22': "#eaca3a", 'X': "#1c8caf"}

    # default colors for chromosomes alternate between black and green
    if colors is None:
        colors = {}
        for i in range(1, 23):
            if i % 2 == 0:
                colors[str(i)] = '#0e6d19'
            else:
                colors[str(i)] = "#000000"
        colors['X'] = "#000000"

    # add global_positions and colors as row fields
    mt = mt.annotate_globals(color_dict=colors)
    locus_expr = mt._fields[locus_field]
    mt = (mt
          .annotate_rows(global_position=locus_expr.global_position(),
                         color=mt.color_dict[locus_expr.contig])
          .drop('color_dict'))

    pval_expr = mt._fields[pval_field]

    # label for hover data must be an array of strings
    # (hl.agg.downsample only takes strings for the label)
    label_expr = hl.array([hl.str(mt.locus),
                           hl.str(mt.alleles),
                           mt.gene,
                           hl.str(pval_expr)])

    # entry boolean for whether p-value is under the threshold for hover data
    mt = (mt.annotate_entries(neg_log_pval=-hl.log(pval_expr),
                              under_threshold=pval_expr < threshold,
                              label=label_expr)
          .key_cols_by(phenotype_field))

    # y-axis range for each phenotype
    mt = (mt.annotate_cols(min_nlp=hl.agg.min(mt.neg_log_pval),
                           max_nlp=hl.agg.max(mt.neg_log_pval)))

    # global position range (x-axis range)
    gp_range = mt.aggregate_rows(
        hl.struct(
            min=hl.agg.min(mt.global_position),
            max=hl.agg.max(mt.global_position)
        ))

    mt = mt.annotate_globals(gp_range=gp_range)

    return mt


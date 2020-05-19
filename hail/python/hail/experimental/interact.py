from IPython.display import display
from ipywidgets import widgets

import hail as hl

from hail.expr.types import summary_type

__all__ = [
    'interact',
]


def interact(obj):
    tab = widgets.Tab()
    base_style = widgets.ButtonStyle()
    selected_style = widgets.ButtonStyle(button_color='#DDFFDD', font_weight='bold')

    if isinstance(obj, hl.Table):
        glob = widgets.Button(description='globals',
                              layout=widgets.Layout(width='150px', height='30px'))
        rows = widgets.Button(description='rows',
                              layout=widgets.Layout(width='150px', height='200px'))
        rows.style = selected_style

        globals_frames = []
        globals_frames.append(widgets.HTML(
            f'<p><big>Global fields, with one value in the dataset.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate_globals", "https://hail.is/docs/0.2/hail.Table.html#hail.Table.annotate_globals")}: '
            f'add new global fields.</li>'
            f'</ul>'
        ))
        append_struct_frames(obj.globals.dtype, globals_frames)

        row_frames = []
        row_frames.append(widgets.HTML(
            f'<p><big>Row fields, with one record per row of the table.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate", "https://hail.is/docs/0.2/hail.Table.html#hail.Table.annotate")}: '
            f'add new fields.</li>'
            f'<li>{html_link("filter", "https://hail.is/docs/0.2/hail.Table.html#hail.Table.filter")}: '
            f'filter rows of the table.</li>'
            f'<li>{html_link("aggregate", "https://hail.is/docs/0.2/hail.Table.html#hail.Table.aggregate")}: '
            f'aggregate over rows to produce a single value.</li>'
            f'</ul>'
        ))
        if len(obj.key) > 0:
            row_frames.append(widgets.HTML(f'<p><big>Key: {list(obj.key)}<big><p>'))
        append_struct_frames(obj.row.dtype, row_frames)

        tab.children = [widgets.VBox(frames) for frames in [globals_frames, row_frames]]
        tab.set_title(0, 'globals')
        tab.set_title(1, 'row')
        tab.selected_index = 1

        box = widgets.VBox([glob, rows])
        buttons = [glob, rows]
    else:
        assert isinstance(obj, hl.MatrixTable)
        glob = widgets.Button(description='globals',
                              layout=widgets.Layout(width='65px', height='30px'))
        cols = widgets.Button(description='cols',
                              layout=widgets.Layout(width='200px', height='30px'))
        rows = widgets.Button(description='rows',
                              layout=widgets.Layout(width='65px', height='200px'))
        entries = widgets.Button(description='entries',
                                 layout=widgets.Layout(width='200px', height='200px'))
        entries.style = selected_style

        globals_frames = []
        globals_frames.append(widgets.HTML(
            f'<p><big>Global fields, with one value in the dataset.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate_globals()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.annotate_globals")}: '
            f'add new global fields.</li>'
            f'</ul>'
        ))
        append_struct_frames(obj.globals.dtype, globals_frames)

        row_frames = []
        row_frames.append(widgets.HTML(
            f'<p><big>Row fields, with one record per row in the dataset.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate_rows()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.annotate_rows")}: '
            f'add new row fields. This method supports {html_link("aggregation", "https://hail.is/docs/0.2/aggregators.html")}, '
            f'aggregating over entries to compute one result per row, e.g. computing the mean depth per variant.</li>'
            f'<li>{html_link("filter_rows()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.filter_rows")}: '
            f'filter rows in the matrix table.</li>'
            f'<li>{html_link("aggregate_rows()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.aggregate_rows")}: '
            f'aggregate over rows (not including entries or columns) to produce a single value, e.g. counting the number of loss-of-function variants.</li>'
            f'<li>{html_link("rows()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.rows")}: '
            f'return the rows as a Hail {html_link("Table", "https://hail.is/docs/0.2/hail.Table.html")}.</li>'
            f'</ul>'
        ))
        if len(obj.row_key) > 0:
            row_frames.append(widgets.HTML(f'<p><big>Row key: {list(obj.row_key)}<big><p>'))
        append_struct_frames(obj.row.dtype, row_frames)

        col_frames = []
        col_frames.append(widgets.HTML(
            f'<p><big>Column fields, with one record per column in the dataset.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate_cols()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.annotate_cols")}: '
            f'add new column fields. This method supports {html_link("aggregation", "https://hail.is/docs/0.2/aggregators.html")}, '
            f'aggregating over entries to compute one result per column, e.g. computing the mean depth per sample.</li>'
            f'<li>{html_link("filter_cols()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.filter_cols")}: '
            f'filter columns in the matrix table.</li>'
            f'<li>{html_link("aggregate_cols()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.aggregate_cols")}: '
            f'aggregate over columns (not including entries or rows) to produce a single value, e.g. counting the number of samples with case status.</li>'
            f'<li>{html_link("cols()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.cols")}: '
            f'return the columns as a Hail {html_link("Table", "https://hail.is/docs/0.2/hail.Table.html")}.'
            f'</li>'
            f'</ul>'
        ))
        if len(obj.col_key) > 0:
            col_frames.append(widgets.HTML(f'<p><big>Column key: {list(obj.col_key)}<big><p>'))
        append_struct_frames(obj.col.dtype, col_frames)

        entry_frames = []
        entry_frames.append(widgets.HTML(
            f'<p><big>Entry fields, with one record per (row, column) pair in the dataset.</big></p>\n'
            f'<p>Commonly used methods:</p>\n'
            f'<ul>'
            f'<li>{html_link("annotate_entries()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.annotate_entries")}: '
            f'add new entry fields.</li>'
            f'<li>{html_link("filter_entries()", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.filter_entries")}: '
            f'filter entries in the matrix table, removing them from downstream operations, like aggregations.</li>'
            f'<li>{html_link("aggregate_entries", "https://hail.is/docs/0.2/hail.MatrixTable.html#hail.MatrixTable.aggregate_entries")}: '
            f'aggregate over entries to produce a single value, e.g. computing mean depth across an entire dataset.</li>'
            f'</ul>'
        ))
        append_struct_frames(obj.entry.dtype, entry_frames)

        tab.children = [widgets.VBox(frames) for frames in [globals_frames, row_frames, col_frames, entry_frames]]
        tab.set_title(0, 'globals')
        tab.set_title(1, 'row')
        tab.set_title(2, 'col')
        tab.set_title(3, 'entry')
        tab.selected_index = 3

        box = widgets.VBox([widgets.HBox([glob, cols]), widgets.HBox([rows, entries])])
        buttons = [glob, rows, cols, entries]

    selection_handler = widgets.IntText(tab.selected_index)
    button_idx = dict(zip(buttons, range(len(buttons))))

    def handle_selection(x):
        if x['name'] == 'value' and x['type'] == 'change':
            buttons[x['old']].style = base_style
            selection = x['new']
            buttons[selection].style = selected_style
            tab.selected_index = selection

    selection_handler.observe(handle_selection)
    widgets.jslink((tab, 'selected_index'), (selection_handler, 'value'))

    def button_action(b):
        selection_handler.value = button_idx[b]

    for button in button_idx:
        button.on_click(button_action)

    display(box, tab)


def html_code(text):
    return f'<pre>{text}</pre>'


def get_type_html(t):
    if isinstance(t, hl.tdict):
        return f'<p>A dictionary mapping keys to values.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" DictExpression", "https://hail.is/docs/0.2/hail.expr.DictExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" CollectionExpression", "https://hail.is/docs/0.2/hail.expr.CollectionExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>\n' \
               f'<p>Access elements using square brackets: {html_code("x[k]")}</p>'
    elif isinstance(t, hl.tset):
        return f'<p>A set of unique values.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" SetExpression", "https://hail.is/docs/0.2/hail.expr.SetExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" CollectionExpression", "https://hail.is/docs/0.2/hail.expr.CollectionExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>\n'
    elif isinstance(t, hl.tarray):
        if hl.expr.types.is_numeric(t.element_type):
            return f'<p>A variable-length array of homogenous numeric values.</p>\n' \
                   f'Documentation:\n<ul>' \
                   f'<li>class: {html_link(" ArrayNumericExpression", "https://hail.is/docs/0.2/hail.expr.ArrayNumericExpression.html")}</li>' \
                   f'<li>inherited class: {html_link(" ArrayExpression", "https://hail.is/docs/0.2/hail.expr.ArrayExpression.html")}</li>' \
                   f'<li>inherited class: {html_link(" CollectionExpression", "https://hail.is/docs/0.2/hail.expr.CollectionExpression.html")}</li>' \
                   f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
                   f'</ul>\n' \
                   f'<p>Access elements using square brackets: {html_code("x[i]")} or slice using Python syntax: {html_code("x[:end], x[start:], x[start:end]")}</p>'
        else:
            return f'<p>A variable-length array of homogenous values.</p>\n' \
                   f'Documentation:\n<ul>' \
                   f'<li>class: {html_link(" ArrayExpression", "https://hail.is/docs/0.2/hail.expr.ArrayExpression.html")}</li>' \
                   f'<li>inherited class: {html_link(" CollectionExpression", "https://hail.is/docs/0.2/hail.expr.CollectionExpression.html")}</li>' \
                   f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
                   f'</ul>\n' \
                   f'<p>Access elements using square brackets: {html_code("x[i]")} or slice using Python syntax: {html_code("x[:end], x[start:], x[start:end]")}</p>'
    elif isinstance(t, hl.tstruct):
        bracket_str = html_code("x[\"foo\"]")
        return f'<p>A structure of named heterogeneous values.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" StructExpression", "https://hail.is/docs/0.2/hail.expr.StructExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>\n' \
               f'<p>Access an element by name with dots or with square brackets: {html_code(f"x.foo, {bracket_str}")}</p>'
    elif isinstance(t, hl.ttuple):
        return f'<p>A 0-indexed tuple of heterogeneous values.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" TupleExpression", "https://hail.is/docs/0.2/hail.expr.TupleExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>\n' \
               f'<p>Access an element using square brackets. For instance, get the first element: {html_code("x[0]")}</p>'
    elif isinstance(t, hl.tinterval):
        return f'<p>An object representing an interval.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" IntervalExpression", "https://hail.is/docs/0.2/hail.expr.IntervalExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>'
    elif isinstance(t, hl.tlocus):
        return f'<p>An object representing a genomic locus (chromomsome and position).</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" LocusExpression", "https://hail.is/docs/0.2/hail.expr.LocusExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Genetics functions", "https://hail.is/docs/0.2/functions/genetics.html")}</li>' \
               f'</ul>'
    elif t == hl.tint32:
        return f'<p>A 32-bit integer.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" Int32Expression", "https://hail.is/docs/0.2/hail.expr.Int32Expression.html")}</li>' \
               f'<li>inherited class: {html_link(" NumericExpression", "https://hail.is/docs/0.2/hail.expr.NumericExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Numeric functions", "https://hail.is/docs/0.2/functions/numeric.html")}</li>' \
               f'<li>{html_link("Statistical functions", "https://hail.is/docs/0.2/functions/stats.html")}</li>' \
               f'</ul>'
    elif t == hl.tint64:
        return f'<p>A 64-bit integer.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" Int64Expression", "https://hail.is/docs/0.2/hail.expr.Int64Expression.html")}</li>' \
               f'<li>inherited class: {html_link(" NumericExpression", "https://hail.is/docs/0.2/hail.expr.NumericExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Numeric functions", "https://hail.is/docs/0.2/functions/numeric.html")}</li>' \
               f'<li>{html_link("Statistical functions", "https://hail.is/docs/0.2/functions/stats.html")}</li>' \
               f'</ul>'
    elif t == hl.tfloat32:
        return f'<p>A 32-bit floating point number.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" Float32Expression", "https://hail.is/docs/0.2/hail.expr.Float32Expression.html")}</li>' \
               f'<li>inherited class: {html_link(" NumericExpression", "https://hail.is/docs/0.2/hail.expr.NumericExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Numeric functions", "https://hail.is/docs/0.2/functions/numeric.html")}</li>' \
               f'<li>{html_link("Statistical functions", "https://hail.is/docs/0.2/functions/stats.html")}</li>' \
               f'</ul>'
    elif t == hl.tfloat64:
        return f'<p>A 64-bit floating point number.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" Float64Expression", "https://hail.is/docs/0.2/hail.expr.Float64Expression.html")}</li>' \
               f'<li>inherited class: {html_link(" NumericExpression", "https://hail.is/docs/0.2/hail.expr.NumericExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Numeric functions", "https://hail.is/docs/0.2/functions/numeric.html")}</li>' \
               f'<li>{html_link("Statistical functions", "https://hail.is/docs/0.2/functions/stats.html")}</li>' \
               f'</ul>'
    elif t == hl.tbool:
        return f'<p>A 64-bit floating point number.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" BooleanExpression", "https://hail.is/docs/0.2/hail.expr.BooleanExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" NumericExpression", "https://hail.is/docs/0.2/hail.expr.NumericExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'<li>{html_link("Numeric functions", "https://hail.is/docs/0.2/functions/numeric.html")}</li>' \
               f'<li>{html_link("Statistical functions", "https://hail.is/docs/0.2/functions/stats.html")}</li>' \
               f'</ul>'

    elif t == hl.tstr:
        return f'<p>A text string.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" StringExpression", "https://hail.is/docs/0.2/hail.expr.StringExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>'
    elif t == hl.tcall:
        return f'<p>An object representing a genotype call.</p>\n' \
               f'Documentation:\n<ul>' \
               f'<li>class: {html_link(" CallExpression", "https://hail.is/docs/0.2/hail.expr.CallExpression.html")}</li>' \
               f'<li>inherited class: {html_link(" Expression", "https://hail.is/docs/0.2/hail.expr.Expression.html")}</li>' \
               f'</ul>'


def html_link(text, dest):
    return f'<a target="_blank" href="{dest}">{text}</a>'


def format_html(s):
    return '<p style="font-family:courier;white-space:pre;line-height: 115%;">{}</p>'.format(
        str(s).replace('<', '&lt').replace('>', '&gt').replace('\n', '</br>'))


def append_struct_frames(t, frames):
    if len(t) == 0:
        frames.append(widgets.HTML('<big>No fields.</big>'))
    else:
        frames.append(widgets.HTML('<big>Fields:</big>'))
    acc = widgets.Accordion([recursive_build(x) for x in t.values()])
    for i, (name, fd) in enumerate(t.items()):
        acc.set_title(i, f'{repr(name)} ({summary_type(fd)})')
    acc.selected_index = None
    frames.append(acc)


def recursive_build(t):
    frames = []

    frames.append(widgets.HTML(get_type_html(t)))

    if isinstance(t, hl.tstruct):
        append_struct_frames(t, frames)
    elif isinstance(t, hl.ttuple):
        if len(t) == 0:
            frames.append(widgets.HTML('<big>No fields.</big>'))
        else:
            frames.append(widgets.HTML('<big>Fields:</big>'))
        acc = widgets.Accordion([recursive_build(x) for x in t.types])
        for i, fd in enumerate(t.types):
            acc.set_title(i, f'[{i}] ({summary_type(fd)})')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(t, (hl.tarray, hl.tset)):
        acc = widgets.Accordion([recursive_build(t.element_type)])
        acc.set_title(0, f'<element> ({summary_type(t.element_type)})')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(t, hl.tdict):
        acc = widgets.Accordion([recursive_build(t.key_type), recursive_build(t.value_type)])
        acc.set_title(0, f'<key> ({summary_type(t.key_type)})')
        acc.set_title(1, f'<value> ({summary_type(t.element_type)})')
        acc.selected_index = None
        frames.append(acc)
    elif isinstance(t, (hl.tinterval)):
        acc = widgets.Accordion([recursive_build(t.point_type)])
        acc.set_title(0, f'<point> ({summary_type(t.point_type)})')
        acc.selected_index = None
        frames.append(acc)

    return widgets.VBox(frames)

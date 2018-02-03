from hail.table import Table

def tablify(expr):
    s = expr._indices.source
    l = len(expr._indices.axes)
    if   l == 0:
        return s.select_globals(_=expr).globals()._
    elif l == 1 and isinstance(s, Table):
        return s.select(_=expr)
    elif l == 1 and s._row_axis in expr._indices.axes:
        return s.select_rows(_=expr).rows_table()
    elif l == 1 and s._col_axis in expr._indices.axes:
        return s.select_cols(_=expr).cols_table()
    elif l == 2:
        return s.select_entries(_=expr)


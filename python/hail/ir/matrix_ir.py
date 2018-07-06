from hail.ir.base_ir import *
from hail.utils.java import escape_str, escape_id

class MatrixAggregateRowsByKey(BaseIR):
    def __init__(self, child, expr):
        super().__init__()
        self.child = child
        self.expr = expr

    def __str__(self):
        '(MatrixAggregateRowsByKey {} {})'.format(self.child, self.expr)

class MatrixRead(BaseIR):
    def __init__(self, path, drop_cols, drop_rows):
        super().__init__()
        self.path = path
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows

    def __str__(self):
        return '(MatrixRead "{}" {} {} None)'.format(
            self.path, self.drop_cols, self.drop_rows)

class MatrixRange(BaseIR):
    def __init__(self, path, n_rows, n_cols, n_partitions):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_partitions = n_partitions

    def __str__(self):
        return '(MatrixRange {} {} {} False False)'.format(
            self.n_rows, self.n_cols,
            self.n_partitions if self.n_partitions else 'None')

class MatrixFilterRows(BaseIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def __str__(self):
        '(MatrixFilterRows {} {})'.format(self.child, self.pred)

class MatrixChooseCols(BaseIR):
    def __init__(self, child, old_entries):
        super().__init__()
        self.child = child
        self.old_entries = old_entries

    def __str__(self):
        '(MatrixChooseCols ({}) {})'.format(
            self.child, ' '.join([str(i) for i in self.old_entries]))

class MatrixMapCols(BaseIR):
    def __init__(self, child, new_col, new_key):
        super().__init__()
        self.child = child
        self.new_col = new_col
        self.new_key = new_key

    def __str__(self):
        '(MatrixMapCols {} {} {})'.format(
            '(' + ' '.join([escape_id(f) for f in self.new_col]) + ')' if self.new_col else 'None',
            self.child, self.new_col)

class MatrixMapEntries(BaseIR):
    def __init__(self, child, new_entry):
        super().__init__()
        self.child = child
        self.new_entry = new_entry

    def __str__(self):
        '(MatrixMapEntries {} {})'.format( self.child, self.new_entry)

class MatrixFilterEntries(BaseIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def __str__(self):
        '(MatrixFilterEntries {} {})'.format(self.child, self.pred)

class MatrixMapRows(BaseIR):
    def __init__(self, child, new_row, new_key):
        super().__init__()
        self.child = child
        self.new_row = new_row
        self.new_key = new_key

    def __str__(self):
        '(MatrixMapEntries {} {} {})'.format(
            '(' + ' '.join([escape_id(f) for (f, _) in self.new_key]) if self.new_key else 'None',
            '(' + ' '.join([escape_id(f) for (_, f) in self.new_key]) if self.new_key else 'None',
            self.child, self.new_row)

class MatrixMapGlobals(BaseIR):
    def __init__(self, child, new_row, value):
        super().__init__()
        self.child = child
        self.new_row = new_row
        self.value = value

    def __str__(self):
        '(MatrixMapGlobals {} {} {})'.format(
            escape_str(json.dumps(self.value)),
            self.child, self.pred)

class MatrixFilterCols(BaseIR):
    def __init__(self, child, pred):
        super().__init__()
        self.child = child
        self.pred = pred

    def __str__(self):
        '(MatrixFilterCols {} {})'.format(self.child, self.pred)

class MatrixCollectColsByKey(BaseIR):
    def __init__(self, child):
        super().__init__()
        self.child = child

    def __str__(self):
        '(MatrixCollectColsByKey {})'.format(self.child)

class MatrixAggregateColsByKey(BaseIR):
    def __init__(self, child, agg_ir):
        super().__init__()
        self.child = child
        self.agg_ir = agg_ir

    def __str__(self):
        '(MatrixAggregateColsByKey {} {})'.format(self.child, self.agg_ir)

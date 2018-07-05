from hail.ir.base_ir import *

class MatrixAggregateRowsByKey(BaseIR):
    def __init__(self, child, expr):
        super().__init__()
        self.child = child
        self.expr = expr

    def __str__(self):
        '(MatrixAggregateRowsByKey {} {})'.format(self.child, self.expr)

class MatrixRead(BaseIR):
    def __init__(self, typ, partition_counts, column_count, drop_cols, drop_rows, reader):
        super().__init__()
        self.typ = typ
        self.partition_counts = partition_counts
        self.column_count = column_count
        self.drop_cols = drop_cols
        self.drop_rows = drop_rows
        self.reader = reader

    def __str__(self):
        '(MatrixRead {} {} {} {} {} {})'.format(
            self.typ.parsableString(),
            '(' + ' '.join([str(n) for n in self.partition_counts]) if self.partition_counts else 'None',
            self.column_count,
            self.drop_cols,
            self.drop_rows,
            self.reader)

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

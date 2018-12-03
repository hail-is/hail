from hail.typecheck import *
from hail.expr.types import tstruct
from hail.utils.java import escape_id

class MatrixType(object):
    @typecheck_method(global_type=tstruct,
                      col_key=sequenceof(str),
                      col_type=tstruct,
                      row_key=sequenceof(str),
                      row_type=tstruct,
                      entry_type=tstruct)
    def __init__(self, global_type, col_key, col_type, row_key,
                 row_type, entry_type):
        assert set(col_key).intersection(set(list(col_type))) == set(col_key)
        assert set(row_key).intersection(set(list(row_type))) == set(row_key)

        self.global_type = global_type
        self.col_key = col_key
        self.col_type = col_type
        self.row_key = row_key
        self.row_type = row_type
        self.entry_type = entry_type

    def __str__(self):
        col_key = ','.join([escape_id(k) for k in self.col_key])
        row_key = ','.join([escape_id(k) for k in self.row_key])
        return f"Matrix{{" \
                f"global:{self.global_type._jtype.parsableString()}," \
                f"col_key:[{col_key}]," \
                f"col:{self.col_type._jtype.parsableString()}," \
                f"row_key:[[{row_key}]]," \
                f"row:{self.row_type._jtype.parsableString()}," \
                f"entry:{self.entry_type._jtype.parsableString()}}}"


class TableType(object):
    @typecheck_method(row_type=tstruct,
                      key=sequenceof(str),
                      global_type=tstruct)
    def __init__(self, row_type, key, global_type):
        assert set(key).intersection(set(list(row_type))) == set(key)

        self.row_type = row_type
        self.key = key
        self.global_type = global_type

    def __str__(self):
        key = ','.join([escape_id(k) for k in self.key])
        return f"Table{{" \
                f"global:{self.global_type._jtype.parsableString()}," \
                f"key:[{key}]," \
                f"row:{self.row_type._jtype.parsableString()}}}"

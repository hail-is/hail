import atexit
import datetime
import difflib
import shutil
import tempfile
from collections import defaultdict, Counter, OrderedDict
from random import Random

import hail
from hail.typecheck import enumeration, typecheck, nullable
from hail.utils.java import Env, joption, error


@typecheck(n_rows=int, n_cols=int, n_partitions=nullable(int))
def range_matrix_table(n_rows, n_cols, n_partitions=None) -> 'hail.MatrixTable':
    """Construct a matrix table with row and column indices and no entry fields.

    Examples
    --------

    >>> range_ds = hl.utils.range_matrix_table(n_rows=100, n_cols=10)

    >>> range_ds.count_rows()
    100

    >>> range_ds.count_cols()
    10

    Notes
    -----
    The resulting matrix table contains the following fields:

     - `row_idx` (:py:data:`.tint32`) - Row index (row key).
     - `col_idx` (:py:data:`.tint32`) - Column index (column key).

    It contains no entry fields.

    This method is meant for testing and learning, and is not optimized for
    production performance.

    Parameters
    ----------
    n_rows : :obj:`int`
        Number of rows.
    n_cols : :obj:`int`
        Number of columns.
    n_partitions : int, optional
        Number of partitions (uses Spark default parallelism if None).

    Returns
    -------
    :class:`.MatrixTable`
    """
    check_positive_and_in_range('range_matrix_table', 'n_rows', n_rows)
    check_positive_and_in_range('range_matrix_table', 'n_cols', n_cols)
    if n_partitions is not None:
        check_positive_and_in_range('range_matrix_table', 'n_partitions', n_partitions)
    return hail.MatrixTable._from_java(Env.hail().variant.MatrixTable.range(Env.hc()._jhc, n_rows, n_cols, joption(n_partitions)))

@typecheck(n=int, n_partitions=nullable(int))
def range_table(n, n_partitions=None) -> 'hail.Table':
    """Construct a table with the row index and no other fields.

    Examples
    --------

    >>> df = hl.utils.range_table(100)

    >>> df.count()
    100

    Notes
    -----
    The resulting table contains one field:

     - `idx` (:py:data:`.tint32`) - Row index (key).

    This method is meant for testing and learning, and is not optimized for
    production performance.

    Parameters
    ----------
    n : int
        Number of rows.
    n_partitions : int, optional
        Number of partitions (uses Spark default parallelism if None).

    Returns
    -------
    :class:`.Table`
    """
    check_positive_and_in_range('range_table', 'n', n)
    if n_partitions is not None:
        check_positive_and_in_range('range_table', 'n_partitions', n_partitions)

    return hail.Table._from_java(Env.hail().table.Table.range(Env.hc()._jhc, n, joption(n_partitions)))

def check_positive_and_in_range(caller, name, value):
    if value <= 0:
        raise ValueError(f"'{caller}': parameter '{name}' must be positive, found {value}")
    elif value > hail.tint32.max_value:
        raise ValueError(f"'{caller}': parameter '{name}' must be less than or equal to {hail.tint32.max_value}, "
                         f"found {value}")

def wrap_to_list(s):
    if isinstance(s, list):
        return s
    else:
        return [s]

def wrap_to_tuple(x):
    if isinstance(x, tuple):
        return x
    else:
        return x,

def wrap_to_sequence(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    else:
        return x,

def get_env_or_default(maybe, envvar, default):
    import os

    return maybe or os.environ.get(envvar) or default


def uri_path(uri):
    return Env.jutils().uriPath(uri)


def local_path_uri(path):
    return 'file://' + path


def new_temp_file(suffix=None, prefix=None, n_char=10):
    return Env.hc()._jhc.getTemporaryFile(n_char, joption(prefix), joption(suffix))


def new_local_temp_dir(suffix=None, prefix=None, dir=None):
    local_temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    atexit.register(shutil.rmtree, local_temp_dir)
    return local_temp_dir


def new_local_temp_file(filename="temp"):
    local_temp_dir = new_local_temp_dir()
    path = local_temp_dir + "/" + filename
    return path


storage_level = enumeration('NONE', 'DISK_ONLY', 'DISK_ONLY_2', 'MEMORY_ONLY',
                            'MEMORY_ONLY_2', 'MEMORY_ONLY_SER', 'MEMORY_ONLY_SER_2',
                            'MEMORY_AND_DISK', 'MEMORY_AND_DISK_2', 'MEMORY_AND_DISK_SER',
                            'MEMORY_AND_DISK_SER_2', 'OFF_HEAP')


def run_command(args):
    import subprocess as sp
    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(e.output)
        raise e


def plural(orig, n, alternate=None):
    if n == 1:
        return orig
    elif alternate:
        return alternate
    else:
        return orig + 's'


def get_obj_metadata(obj):
    from hail.matrixtable import MatrixTable, GroupedMatrixTable
    from hail.table import Table, GroupedTable
    from hail.utils import Struct
    from hail.expr.expressions import StructExpression

    def table_error(index_obj):
        def fmt_field(field):
            assert field in index_obj._fields
            inds = index_obj[field]._indices
            if inds == index_obj._global_indices:
                return "'{}' [globals]".format(field)
            elif inds == index_obj._row_indices:
                return "'{}' [row]".format(field)
            elif inds == index_obj._col_indices:  # Table will never get here
                return "'{}' [col]".format(field)
            else:
                assert inds == index_obj._entry_indices
                return "'{}' [entry]".format(field)
        return fmt_field

    def struct_error(s):
        def fmt_field(field):
            assert field in s._fields
            return "'{}'".format(field)
        return fmt_field

    if isinstance(obj, MatrixTable):
        return 'MatrixTable', MatrixTable, table_error(obj), True
    elif isinstance(obj, GroupedMatrixTable):
        return 'GroupedMatrixTable', GroupedMatrixTable, table_error(obj._parent), True
    elif isinstance(obj, Table):
        return 'Table', Table, table_error(obj), True
    elif isinstance(obj, GroupedTable):
        return 'GroupedTable', GroupedTable, table_error(obj), False
    elif isinstance(obj, Struct):
        return 'Struct', Struct, struct_error(obj), False
    elif isinstance(obj, StructExpression):
        return 'StructExpression', StructExpression, struct_error(obj), True
    else:
        raise NotImplementedError(obj)


def get_nice_attr_error(obj, item):
    class_name, cls, handler, has_describe = get_obj_metadata(obj)

    if item.startswith('_'):
        # don't handle 'private' attribute access
        return "{} instance has no attribute '{}'".format(class_name, item)
    else:
        field_names = obj._fields.keys()
        field_dict = defaultdict(lambda: [])
        for f in field_names:
            field_dict[f.lower()].append(f)

        obj_namespace = {x for x in dir(cls) if not x.startswith('_')}
        inherited = {x for x in obj_namespace if x not in cls.__dict__}
        methods = {x for x in obj_namespace if x in cls.__dict__ and callable(cls.__dict__[x])}
        props = obj_namespace - methods - inherited

        item_lower = item.lower()

        field_matches = difflib.get_close_matches(item_lower, field_dict, n=5)
        inherited_matches = difflib.get_close_matches(item_lower, inherited, n=5)
        method_matches = difflib.get_close_matches(item_lower, methods, n=5)
        prop_matches = difflib.get_close_matches(item_lower, props, n=5)

        s = ["{} instance has no field, method, or property '{}'".format(class_name, item)]
        if any([field_matches, method_matches, prop_matches, inherited_matches]):
            s.append('\n    Did you mean:')
            if field_matches:
                l = []
                for f in field_matches:
                    l.extend(field_dict[f])
                word = plural('field', len(l))
                s.append('\n        Data {}: {}'.format(word, ', '.join(handler(f) for f in l)))
            if method_matches:
                word = plural('method', len(method_matches))
                s.append('\n        {} {}: {}'.format(class_name, word,
                                                      ', '.join("'{}'".format(m) for m in method_matches)))
            if prop_matches:
                word = plural('property', len(prop_matches), 'properties')
                s.append('\n        {} {}: {}'.format(class_name, word,
                                                      ', '.join("'{}'".format(p) for p in prop_matches)))
            if inherited_matches:
                word = plural('inherited method', len(inherited_matches))
                s.append('\n        {} {}: {}'.format(class_name, word,
                                                      ', '.join("'{}'".format(m) for m in inherited_matches)))
        elif has_describe:
            s.append("\n    Hint: use 'describe()' to show the names of all data fields.")
        return ''.join(s)


def get_nice_field_error(obj, item):
    class_name, _, handler, has_describe = get_obj_metadata(obj)

    field_names = obj._fields.keys()
    dd = defaultdict(lambda: [])
    for f in field_names:
        dd[f.lower()].append(f)

    item_lower = item.lower()

    field_matches = difflib.get_close_matches(item_lower, dd, n=5)

    s = ["{} instance has no field '{}'".format(class_name, item)]
    if field_matches:
        s.append('\n    Did you mean:')
        for f in field_matches:
            for orig_f in dd[f]:
                s.append("\n        {}".format(handler(orig_f)))
    if has_describe:
        s.append("\n    Hint: use 'describe()' to show the names of all data fields.")
    return ''.join(s)

def check_collisions(fields, name, indices):
    from hail.expr.expressions import ExpressionException
    if name in fields and not fields[name]._indices == indices:
        msg = "name collision with field indexed by {}: {}".format(list(fields[name]._indices.axes), repr(name))
        error('Analysis exception: {}'.format(msg))
        raise ExpressionException(msg)

def check_field_uniqueness(fields):
    for k, v in Counter(fields).items():
        if v > 1:
            from hail.expr.expressions import ExpressionException
            raise ExpressionException("selection would produce duplicate field '{}'".format(repr(k)))

def check_keys(name, indices):
    from hail.expr.expressions import ExpressionException
    if indices.key is None:
        return
    if name in set(indices.key):
        msg = "cannot overwrite key field {} with annotate, select or drop; use key_by to modify keys.".format(repr(name))
        error('Analysis exception: {}'.format(msg))
        raise ExpressionException(msg)

def get_select_exprs(caller, exprs, named_exprs, indices, protect_keys=True):
    from hail.expr.expressions import to_expr, ExpressionException, analyze
    exprs = [to_expr(e) if not isinstance(e, str) else indices.source[e] for e in exprs]
    named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
    assignments = OrderedDict()

    for e in exprs:
        if not e._ir.is_nested_field:
            raise ExpressionException("method '{}' expects keyword arguments for complex expressions".format(caller))
        analyze(caller, e, indices, broadcast=False)
        if protect_keys:
            check_keys(e._ir.name, indices)
        assignments[e._ir.name] = e
    for k, e in named_exprs.items():
        if protect_keys:
            check_keys(k, indices)
        check_collisions(indices.source._fields, k, indices)
        assignments[k] = e
    check_field_uniqueness(assignments.keys())
    return assignments

def get_annotate_exprs(caller, named_exprs, indices):
    from hail.expr.expressions import to_expr, ExpressionException
    named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
    for k, v in named_exprs.items():
        check_keys(k, indices)
        if indices.key and k in indices.key.keys():
            raise ExpressionException("'{}' cannot overwrite key field: {}"
                                      .format(caller, repr(k)))
        check_collisions(indices.source._fields, k, indices)
    return named_exprs

def process_joins(obj, exprs):
    all_uids = []
    left = obj
    used_joins = set()

    for e in exprs:
        joins = e._ir.search(lambda a: isinstance(a, hail.ir.Join))
        for j in sorted(joins, key=lambda j: j.idx): # Make sure joins happen in order
            if j not in used_joins:
                left = j.join_func(left)
                all_uids.extend(j.temp_vars)
                used_joins.add(j)

    def cleanup(table):
        remaining_uids = [uid for uid in all_uids if uid in table._fields]
        return table.drop(*remaining_uids)

    return left, cleanup

def divide_null(num, denom):
    from hail.expr.expressions.base_expression import unify_types_limited
    from hail.expr import null, cond
    typ = unify_types_limited(num.dtype, denom.dtype)
    assert typ is not None
    return cond(denom != 0, num / denom, null(typ))


class HailSeedGenerator(object):
    def __init__(self, seed):
        self.seed = seed
        self.generator = Random(seed)

    def set_seed(self, seed):
        self.__init__(seed)

    def next_seed(self):
        return self.generator.randint(0, (1 << 63) - 1)


def timestamp_path(base, suffix=''):
    return ''.join([base,
                    '-',
                    datetime.datetime.now().strftime("%Y%m%d-%H%M"),
                    suffix])
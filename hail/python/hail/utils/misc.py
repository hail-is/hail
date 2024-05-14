import atexit
import datetime
import difflib
import json
import os
import re
import secrets
import shutil
import string
import tempfile
from collections import defaultdict, Counter
from contextlib import contextmanager
from io import StringIO
from typing import Optional, Literal
from urllib.parse import urlparse

import hail
import hail as hl
from hail.typecheck import enumeration, typecheck, nullable
from hail.utils.java import Env, error


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
    check_nonnegative_and_in_range('range_matrix_table', 'n_rows', n_rows)
    check_nonnegative_and_in_range('range_matrix_table', 'n_cols', n_cols)
    if n_partitions is not None:
        check_positive_and_in_range('range_matrix_table', 'n_partitions', n_partitions)
    return hail.MatrixTable(
        hail.ir.MatrixRead(
            hail.ir.MatrixRangeReader(n_rows, n_cols, n_partitions),
            _assert_type=hl.tmatrix(
                hl.tstruct(),
                hl.tstruct(col_idx=hl.tint32),
                ['col_idx'],
                hl.tstruct(row_idx=hl.tint32),
                ['row_idx'],
                hl.tstruct(),
            ),
        )
    )


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
    check_nonnegative_and_in_range('range_table', 'n', n)
    if n_partitions is not None:
        check_positive_and_in_range('range_table', 'n_partitions', n_partitions)

    return hail.Table(hail.ir.TableRange(n, n_partitions))


def check_positive_and_in_range(caller, name, value):
    if value <= 0:
        raise ValueError(f"'{caller}': parameter '{name}' must be positive, found {value}")
    elif value > hail.tint32.max_value:
        raise ValueError(
            f"'{caller}': parameter '{name}' must be less than or equal to {hail.tint32.max_value}, " f"found {value}"
        )


def check_nonnegative_and_in_range(caller, name, value):
    if value < 0:
        raise ValueError(f"'{caller}': parameter '{name}' must be non-negative, found {value}")
    elif value > hail.tint32.max_value:
        raise ValueError(
            f"'{caller}': parameter '{name}' must be less than or equal to {hail.tint32.max_value}, " f"found {value}"
        )


def wrap_to_list(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, tuple):
        return list(s)
    else:
        return [s]


def wrap_to_tuple(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x,)


def wrap_to_sequence(x):
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def get_env_or_default(maybe, envvar, default):
    import os

    return maybe or os.environ.get(envvar) or default


def uri_path(uri):
    return urlparse(uri).path


def local_path_uri(path):
    return 'file://' + path


def new_temp_file(prefix=None, extension=None):
    tmpdir = Env.hc()._tmpdir
    alphabet = string.ascii_letters + string.digits
    token = ''.join([secrets.choice(alphabet) for _ in range(22)])

    f = token
    if prefix is not None:
        f = f'{prefix}-{f}'
    if extension is not None:
        f = f'{f}.{extension}'
    return f'{tmpdir}/{f}'


def new_local_temp_dir(suffix=None, prefix=None, dir=None):
    local_temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    atexit.register(shutil.rmtree, local_temp_dir)
    return local_temp_dir


def new_local_temp_file(filename: str = 'temp') -> str:
    local_temp_dir = new_local_temp_dir()
    path = local_temp_dir + "/" + filename
    return path


@contextmanager
def with_local_temp_file(filename: str = 'temp') -> str:
    path = new_local_temp_file(filename)
    try:
        yield path
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


storage_level = enumeration(
    'NONE',
    'DISK_ONLY',
    'DISK_ONLY_2',
    'MEMORY_ONLY',
    'MEMORY_ONLY_2',
    'MEMORY_ONLY_SER',
    'MEMORY_ONLY_SER_2',
    'MEMORY_AND_DISK',
    'MEMORY_AND_DISK_2',
    'MEMORY_AND_DISK_SER',
    'MEMORY_AND_DISK_SER_2',
    'OFF_HEAP',
)


def run_command(args):
    import subprocess as sp

    try:
        sp.check_output(args, stderr=sp.STDOUT)
    except sp.CalledProcessError as e:
        print(e.output)
        raise e


def hl_plural(orig, n, alternate=None):
    if alternate is None:
        plural = orig + 's'
    else:
        plural = alternate
    return hl.if_else(n == 1, orig, plural)


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
    from hail.expr.expressions import StructExpression, ArrayStructExpression, SetStructExpression

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
    elif isinstance(obj, ArrayStructExpression):
        return 'ArrayStructExpression', ArrayStructExpression, struct_error(obj), True
    elif isinstance(obj, SetStructExpression):
        return 'SetStructExpression', SetStructExpression, struct_error(obj), True
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
                fs = []
                for f in field_matches:
                    fs.extend(field_dict[f])
                word = plural('field', len(fs))
                s.append('\n        Data {}: {}'.format(word, ', '.join(handler(f) for f in fs)))
            if method_matches:
                word = plural('method', len(method_matches))
                s.append(
                    '\n        {} {}: {}'.format(class_name, word, ', '.join("'{}'".format(m) for m in method_matches))
                )
            if prop_matches:
                word = plural('property', len(prop_matches), 'properties')
                s.append(
                    '\n        {} {}: {}'.format(class_name, word, ', '.join("'{}'".format(p) for p in prop_matches))
                )
            if inherited_matches:
                word = plural('inherited method', len(inherited_matches))
                s.append(
                    '\n        {} {}: {}'.format(
                        class_name, word, ', '.join("'{}'".format(m) for m in inherited_matches)
                    )
                )
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


def check_collisions(caller, names, indices, override_protected_indices=None):
    from hail.expr.expressions import ExpressionException

    fields = indices.source._fields

    if override_protected_indices is not None:

        def invalid(e):
            return e._indices in override_protected_indices

    else:

        def invalid(e):
            return e._indices != indices

    # check collisions with fields on other axes
    for name in names:
        if name in fields and invalid(fields[name]):
            msg = f"{caller!r}: name collision with field indexed by {list(fields[name]._indices.axes)}: {name!r}"
            error('Analysis exception: {}'.format(msg))
            raise ExpressionException(msg)

    # check duplicate fields
    for k, v in Counter(names).items():
        if v > 1:
            from hail.expr.expressions import ExpressionException

            raise ExpressionException(f"{caller!r}: selection would produce duplicate field {k!r}")


def get_key_by_exprs(caller, exprs, named_exprs, indices, override_protected_indices=None):
    from hail.expr.expressions import to_expr, ExpressionException, analyze

    exprs = [indices.source[e] if isinstance(e, str) else e for e in exprs]
    named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

    bindings = []

    def is_top_level_field(e):
        return e in indices.source._fields_inverse

    existing_key_fields = []
    final_key = []
    for e in exprs:
        analyze(caller, e, indices, broadcast=False)
        if not e._ir.is_nested_field:
            raise ExpressionException(
                f"{caller!r} expects keyword arguments for complex expressions\n"
                f"  Correct:   ht = ht.key_by('x')\n"
                f"  Correct:   ht = ht.key_by(ht.x)\n"
                f"  Correct:   ht = ht.key_by(x = ht.x.replace(' ', '_'))\n"
                f"  INCORRECT: ht = ht.key_by(ht.x.replace(' ', '_'))"
            )

        name = e._ir.name
        final_key.append(name)

        if not is_top_level_field(e):
            bindings.append((name, e))
        else:
            existing_key_fields.append(name)

    final_key.extend(named_exprs)
    bindings.extend(named_exprs.items())
    check_collisions(caller, final_key, indices, override_protected_indices=override_protected_indices)
    return final_key, dict(bindings)


def check_keys(caller, name, protected_key):
    from hail.expr.expressions import ExpressionException

    if name in protected_key:
        msg = (
            f"{caller!r}: cannot overwrite key field {name!r} with annotate, select or drop; "
            f"use key_by to modify keys."
        )
        error('Analysis exception: {}'.format(msg))
        raise ExpressionException(msg)


def get_select_exprs(caller, exprs, named_exprs, indices, base_struct):
    from hail.expr.expressions import to_expr, ExpressionException, analyze

    exprs = [indices.source[e] if isinstance(e, str) else e for e in exprs]
    named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
    select_fields = indices.protected_key[:]
    protected_key = set(select_fields)
    insertions = {}

    final_fields = select_fields[:]

    def is_top_level_field(e):
        return e in indices.source._fields_inverse

    for e in exprs:
        if not e._ir.is_nested_field:
            raise ExpressionException(
                f"{caller!r} expects keyword arguments for complex expressions\n"
                f"  Correct:   ht = ht.select('x')\n"
                f"  Correct:   ht = ht.select(ht.x)\n"
                f"  Correct:   ht = ht.select(x = ht.x.replace(' ', '_'))\n"
                f"  INCORRECT: ht = ht.select(ht.x.replace(' ', '_'))"
            )
        analyze(caller, e, indices, broadcast=False)

        name = e._ir.name
        check_keys(caller, name, protected_key)
        final_fields.append(name)
        if is_top_level_field(e):
            select_fields.append(name)
        else:
            insertions[name] = e
    for k, e in named_exprs.items():
        check_keys(caller, k, protected_key)
        final_fields.append(k)
        insertions[k] = e

    check_collisions(caller, final_fields, indices)

    if final_fields == select_fields + list(insertions):
        # don't clog the IR with redundant field names
        s = base_struct.select(*select_fields).annotate(**insertions)
    else:
        s = base_struct.select(*select_fields)._annotate_ordered(insertions, final_fields)

    assert list(s) == final_fields
    return s


def check_annotate_exprs(caller, named_exprs, indices, agg_axes):
    from hail.expr.expressions import analyze

    protected_key = set(indices.protected_key)
    for k, v in named_exprs.items():
        analyze(f'{caller}: field {k!r}', v, indices, agg_axes, broadcast=True)
        check_keys(caller, k, protected_key)
    check_collisions(caller, list(named_exprs), indices)
    return named_exprs


def process_joins(obj, exprs):
    all_uids = []
    left = obj
    used_joins = set()

    for e in exprs:
        joins = e._ir.search(lambda a: isinstance(a, hail.ir.Join))
        for j in sorted(joins, key=lambda j: j.idx):  # Make sure joins happen in order
            if j.idx not in used_joins:
                left = j.join_func(left)
                all_uids.extend(j.temp_vars)
                used_joins.add(j.idx)

    def cleanup(table):
        remaining_uids = [uid for uid in all_uids if uid in table._fields]
        return table.drop(*remaining_uids)

    return left, cleanup


def divide_null(num, denom):
    from hail.expr.expressions.base_expression import unify_types_limited
    from hail.expr import missing, if_else

    typ = unify_types_limited(num.dtype, denom.dtype)
    assert typ is not None
    return if_else(denom != 0, num / denom, missing(typ))


def lookup_bit(byte, which_bit):
    return (byte >> which_bit) & 1


def timestamp_path(base, suffix=''):
    return ''.join([base, '-', datetime.datetime.now().strftime("%Y%m%d-%H%M"), suffix])


def upper_hex(n, num_digits=None):
    if num_digits is None:
        return "{0:X}".format(n)
    else:
        return "{0:0{1}X}".format(n, num_digits)


def escape_str(s, backticked=False):
    sb = StringIO()

    rewrite_dict = {'\b': '\\b', '\n': '\\n', '\t': '\\t', '\f': '\\f', '\r': '\\r'}

    for ch in s:
        chNum = ord(ch)
        if chNum > 0x7F:
            sb.write("\\u" + upper_hex(chNum, 4))
        elif chNum < 32:
            if ch in rewrite_dict:
                sb.write(rewrite_dict[ch])
            else:
                if chNum > 0xF:
                    sb.write("\\u00" + upper_hex(chNum))
                else:
                    sb.write("\\u000" + upper_hex(chNum))
        else:
            if ch == '"':
                if backticked:
                    sb.write('"')
                else:
                    sb.write('\\"')
            elif ch == '`':
                if backticked:
                    sb.write("\\`")
                else:
                    sb.write("`")
            elif ch == '\\':
                sb.write('\\\\')
            else:
                sb.write(ch)

    escaped = sb.getvalue()
    sb.close()

    return escaped


def escape_id(s):
    if re.fullmatch(r'[_a-zA-Z]\w*', s):
        return s
    else:
        return "`{}`".format(escape_str(s, backticked=True))


def parsable_strings(strs):
    strs = ' '.join(f'"{escape_str(s)}"' for s in strs)
    return f"({strs})"


def _dumps_partitions(partitions, row_key_type):
    parts_type = partitions.dtype
    if not (isinstance(parts_type, hl.tarray) and isinstance(parts_type.element_type, hl.tinterval)):
        raise ValueError(f'partitions type invalid: {parts_type} must be array of intervals')

    point_type = parts_type.element_type.point_type

    f1, t1 = next(iter(row_key_type.items()))
    if point_type == t1:
        partitions = hl.map(
            lambda x: hl.interval(
                start=hl.struct(**{f1: x.start}),
                end=hl.struct(**{f1: x.end}),
                includes_start=x.includes_start,
                includes_end=x.includes_end,
            ),
            partitions,
        )
    else:
        if not isinstance(point_type, hl.tstruct):
            raise ValueError(f'partitions has wrong type: {point_type} must be struct or type of first row key field')
        if not point_type._is_prefix_of(row_key_type):
            raise ValueError(f'partitions type invalid: {point_type} must be prefix of {row_key_type}')

    s = json.dumps(partitions.dtype._convert_to_json(hl.eval(partitions)))
    return s, partitions.dtype


def default_handler():
    try:
        from IPython.display import display

        return display
    except ImportError:
        return print


def guess_cloud_spark_provider() -> Optional[Literal['dataproc', 'hdinsight']]:
    if 'HAIL_DATAPROC' in os.environ:
        return 'dataproc'
    if 'AZURE_SPARK' in os.environ or 'hdinsight' in os.getenv('CLASSPATH', ''):
        return 'hdinsight'
    return None


def no_service_backend(unsupported_feature):
    from hail import current_backend
    from hail.backend.service_backend import ServiceBackend

    if isinstance(current_backend(), ServiceBackend):
        raise NotImplementedError(
            f'{unsupported_feature!r} is not yet supported on the service backend.'
            f'\n  If this is a pressing need, please alert the team on the discussion'
            f'\n  forum to aid in prioritization: https://discuss.hail.is'
        )


ANY_REGION = ['any_region']

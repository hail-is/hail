from hail.utils.java import handle_py4j, Env, joption
from hail.typecheck import enumeration
import difflib
from collections import defaultdict


class FunctionDocumentation(object):
    @handle_py4j
    def types_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeTypesDocs(file_name)

    @handle_py4j
    def functions_rst(self, file_name):
        Env.hail().utils.FunctionDocumentation.makeFunctionsDocs(file_name)


def wrap_to_list(s):
    if isinstance(s, list):
        return s
    else:
        return [s]


def get_env_or_default(maybe, envvar, default):
    import os

    return maybe or os.environ.get(envvar) or default


def get_URI(path):
    return Env.jutils().getURI(path)


@handle_py4j
def new_temp_file(n_char=10, prefix=None, suffix=None):
    return Env.hc()._jhc.getTemporaryFile(n_char, joption(prefix), joption(suffix))


storage_level = enumeration('NONE', 'DISK_ONLY', 'DISK_ONLY_2', 'MEMORY_ONLY',
                            'MEMORY_ONLY_2', 'MEMORY_ONLY_SER', 'MEMORY_ONLY_SER_2',
                            'MEMORY_AND_DISK', 'MEMORY_AND_DISK_2', 'MEMORY_AND_DISK_SER',
                            'MEMORY_AND_DISK_SER_2', 'OFF_HEAP')


def plural(orig, n, alternate=None):
    if n == 1:
        return orig
    elif alternate:
        return alternate
    else:
        return orig + 's'


def get_nice_attr_error(obj, item):
    from hail.api2 import MatrixTable, GroupedMatrixTable, Table, GroupedTable
    if isinstance(obj, MatrixTable):
        class_name = 'MatrixTable'
        cls = MatrixTable
        index_obj = obj
    elif isinstance(obj, GroupedMatrixTable):
        class_name = 'GroupedMatrixTable'
        cls = GroupedMatrixTable
        index_obj = obj._parent
    elif isinstance(obj, Table):
        class_name = 'Table'
        cls = Table
        index_obj = obj
    elif isinstance(obj, GroupedTable):
        class_name = 'GroupedTable'
        cls = GroupedTable
        index_obj = obj._parent
    else:
        raise NotImplementedError(obj)

    if item.startswith('_'):
        return "{} instance has no attribute '{}'".format(class_name, item)
    else:
        fields = obj._fields.keys()
        dd = defaultdict(lambda: [])
        for f in fields:
            dd[f.lower()].append(f)

        obj_namespace = {x for x in dir(cls) if not x.startswith('_')}
        methods = {x for x in obj_namespace if callable(getattr(obj, x))}
        props = obj_namespace - methods

        item_lower = item.lower()

        field_matches = difflib.get_close_matches(item_lower, dd, n=5)
        method_matches = difflib.get_close_matches(item_lower, methods, n=5)
        prop_matches = difflib.get_close_matches(item_lower, props, n=5)

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

        s = ["{} instance has no field, method, or property '{}'".format(class_name, item)]
        if any([field_matches, method_matches, prop_matches]):
            s.append('\n    Did you mean:')
            if field_matches:
                l = []
                for f in field_matches:
                    l.extend(dd[f])
                word = plural('field', len(l))
                s.append('\n        Data {}: {}'.format(word, ', '.join(fmt_field(f) for f in l)))
            if method_matches:
                word = plural('method', len(method_matches))
                s.append('\n        {} {}: {}'.format(class_name, word,
                                                      ', '.join("'{}'".format(m) for m in method_matches)))
            if prop_matches:
                word = plural('property', len(prop_matches), 'properties')
                s.append('\n        {} {}: {}'.format(class_name, word,
                                                      ', '.join("'{}'".format(p) for p in prop_matches)))
        else:
            s.append("\n    Hint: use 'describe()' to show the names of all data fields.")
        return ''.join(s)


def get_nice_field_error(obj, item):
    from hail.api2 import MatrixTable, GroupedMatrixTable, Table, GroupedTable
    from hail.expr import TStruct
    if isinstance(obj, MatrixTable):
        class_name = 'MatrixTable'
        index_obj = obj
    elif isinstance(obj, GroupedMatrixTable):
        class_name = 'GroupedMatrixTable'
        index_obj = obj._parent
    elif isinstance(obj, Table):
        class_name = 'Table'
        index_obj = obj
    elif isinstance(obj, GroupedTable):
        class_name = 'GroupedTable'
        index_obj = obj._parent
    else:
        raise NotImplementedError(obj)

    fields = obj._fields.keys()
    dd = defaultdict(lambda: [])
    for f in fields:
        dd[f.lower()].append(f)

    def fmt_field(field):
        assert field in index_obj._fields
        field_expr = index_obj[field]
        inds = field_expr._indices
        if inds == index_obj._global_indices:
            return "'{}' [global field]".format(field)
        elif inds == index_obj._row_indices:
            return "'{}' [row field]".format(field)
        elif inds == index_obj._col_indices:  # Table will never get here
            return "'{}' [col field]".format(field)
        else:
            assert inds == index_obj._entry_indices
            return "'{}' [entry field]".format(field)

    item_lower = item.lower()

    field_matches = difflib.get_close_matches(item_lower, dd, n=5)

    s = ["{} instance has no field '{}'".format(class_name, item)]
    if field_matches:
        s.append('\n    Did you mean:')
        for f in field_matches:
            for orig_f in dd[f]:
                s.append("\n        {}".format(fmt_field(orig_f)))
    s.append("\n    Hint: use 'describe()' to show the names of all data fields.")
    return ''.join(s)

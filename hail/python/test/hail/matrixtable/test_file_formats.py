import unittest

import hail as hl
from hail.utils.java import Env, scala_object
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def create_backward_compatibility_files():
    import os

    all_values_table, all_values_matrix_table = create_all_values_datasets()

    file_version = Env.hail().expr.ir.FileFormat.version().toString()
    supported_codecs = scala_object(Env.hail().io, 'BufferSpec').specs()

    table_dir = resource(os.path.join('backward_compatability', str(file_version), 'table'))
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    matrix_table_dir = resource(os.path.join('backward_compatability', str(file_version), 'matrix_table'))
    if not os.path.exists(matrix_table_dir):
        os.makedirs(matrix_table_dir)

    i = 0
    for codec in supported_codecs:
        all_values_table.write(os.path.join(table_dir, f'{i}.ht'), overwrite=True, _codec_spec=codec.toString())
        all_values_matrix_table.write(os.path.join(matrix_table_dir, f'{i}.hmt'), overwrite=True,
                                      _codec_spec=codec.toString())
        i += 1


class Tests(unittest.TestCase):
    @unittest.skip  # comment this line to generate files for new versions
    def test_write(self):
        create_backward_compatibility_files()

    def test_backward_compatability(self):
        import os

        def backward_compatible_same(current, old):
            if isinstance(current, hl.Table):
                current = current.select_globals(*old.globals)
                current = current.select(*old.row_value)
            else:
                current = current.select_globals(*old.globals)
                current = current.select_rows(*old.row_value)
                current = current.select_cols(*old.col_value)
                current = current.select_entries(*old.entry)
            return current._same(old)

        all_values_table, all_values_matrix_table = create_all_values_datasets()

        resource_dir = resource('backward_compatability')
        fs = hl.current_backend().fs
        versions = [os.path.basename(x['path']) for x in fs.ls(resource_dir)]

        n = 0
        for v in versions:
            table_dir = os.path.join(resource_dir, v, 'table')
            i = 0
            f = os.path.join(table_dir, '{}.ht'.format(i))
            while fs.exists(f):
                ds = hl.read_table(f)
                assert backward_compatible_same(all_values_table, ds)
                i += 1
                f = os.path.join(table_dir, '{}.ht'.format(i))
                n += 1

            matrix_table_dir = os.path.join(resource_dir, v, 'matrix_table')
            i = 0
            f = os.path.join(matrix_table_dir, '{}.hmt'.format(i))
            while fs.exists(f):
                ds = hl.read_matrix_table(f)
                assert backward_compatible_same(all_values_matrix_table, ds)
                i += 1
                f = os.path.join(matrix_table_dir, '{}.hmt'.format(i))
                n += 1

        assert n == 72

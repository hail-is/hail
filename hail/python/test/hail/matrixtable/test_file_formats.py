import unittest

import hail as hl
from hail.utils.java import Env, scala_object
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def create_backward_compatibility_files():
    import os

    all_values_table, all_values_matrix_table = create_all_values_datasets()

    file_version = Env.hail().variant.FileFormat.version().toString()
    supported_codecs = scala_object(Env.hail().io, 'CodecSpec').supportedCodecSpecs()

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

        all_values_table, all_values_matrix_table = create_all_values_datasets()

        table_dir = resource('backward_compatability/1.0.0/table')
        matrix_table_dir = resource('backward_compatability/1.0.0/matrix_table')

        n = 0
        i = 0
        f = os.path.join(table_dir, '{}.ht'.format(i))
        while os.path.exists(f):
            ds = hl.read_table(f)
            self.assertTrue(ds._same(all_values_table))
            i += 1
            f = os.path.join(table_dir, '{}.ht'.format(i))
            n += 1

        i = 0
        f = os.path.join(matrix_table_dir, '{}.hmt'.format(i))
        while os.path.exists(f):
            ds = hl.read_matrix_table(f)
            self.assertTrue(ds._same(all_values_matrix_table))
            i += 1
            f = os.path.join(matrix_table_dir, '{}.hmt'.format(i))
            n += 1

        self.assertEqual(n, 8)

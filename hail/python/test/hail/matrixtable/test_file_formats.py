import unittest

import hail as hl
from hail.utils.java import Env, scala_object
from ..helpers import *

setUpModule = startTestHailContext
tearDownModule = stopTestHailContext


def create_all_values_datasets():
    all_values = hl.struct(
        f32=hl.float32(3.14),
        i64=hl.int64(-9),
        m=hl.null(hl.tfloat64),
        astruct=hl.struct(a=hl.null(hl.tint32), b=5.5),
        mstruct=hl.null(hl.tstruct(x=hl.tint32, y=hl.tstr)),
        aset=hl.set(['foo', 'bar', 'baz']),
        mset=hl.null(hl.tset(hl.tfloat64)),
        d=hl.dict({hl.array(['a', 'b']): 0.5, hl.array(['x', hl.null(hl.tstr), 'z']): 0.3}),
        md=hl.null(hl.tdict(hl.tint32, hl.tstr)),
        h38=hl.locus('chr22', 33878978, 'GRCh38'),
        ml=hl.null(hl.tlocus('GRCh37')),
        i=hl.interval(
            hl.locus('1', 999),
            hl.locus('1', 1001)),
        c=hl.call(0, 1),
        mc=hl.null(hl.tcall),
        t=hl.tuple([hl.call(1, 2, phased=True), 'foo', hl.null(hl.tstr)]),
        mt=hl.null(hl.ttuple(hl.tlocus('GRCh37'), hl.tbool)))

    def prefix(s, p):
        return hl.struct(**{p + k: s[k] for k in s})

    all_values_table = (hl.utils.range_table(5, n_partitions=3)
                        .annotate_globals(**prefix(all_values, 'global_'))
                        .annotate(**all_values)
                        .cache())

    all_values_matrix_table = (hl.utils.range_matrix_table(3, 2, n_partitions=2)
                               .annotate_globals(**prefix(all_values, 'global_'))
                               .annotate_rows(**prefix(all_values, 'row_'))
                               .annotate_cols(**prefix(all_values, 'col_'))
                               .annotate_entries(**prefix(all_values, 'entry_'))
                               .cache())

    return all_values_table, all_values_matrix_table


def create_backward_compatibility_files():
    import os

    all_values_table, all_values_matrix_table = create_all_values_datasets()

    file_version = Env.hail().variant.FileFormat.version().toString()
    supported_codecs = scala_object(Env.hail().io, 'CodecSpec').supportedCodecSpecs()

    table_dir = 'backward_compatability/{}/table'.format(file_version)
    if not os.path.exists(table_dir):
        os.makedirs(table_dir)

    matrix_table_dir = 'backward_compatability/{}/matrix_table'.format(file_version)
    if not os.path.exists(matrix_table_dir):
        os.makedirs(matrix_table_dir)

    i = 0
    for codec in supported_codecs:
        all_values_table.write('{}/{}.ht'.format(table_dir, i), overwrite=True, _codec_spec=codec.toString())
        all_values_matrix_table.write('{}/{}.hmt'.format(matrix_table_dir, i), overwrite=True,
                                      _codec_spec=codec.toString())
        i += 1


class Tests(unittest.TestCase):
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

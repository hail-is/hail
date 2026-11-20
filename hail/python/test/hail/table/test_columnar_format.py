"""Conformance tests for tables written in the columnar format.

These read Hail's output with pyarrow, independently of Hail, and check it against the columnar
format specification (``dev-docs/hail-query/columnar-format-spec.md``).
"""

import base64
import io
import secrets
from itertools import pairwise
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hail as hl
import hailtop.fs as hfs
from hail import ir
from hail.utils import FatalError, HailUserError, new_temp_file
from hail.utils.java import Env
from hail.utils.misc import is_committed

from ..columnar_helpers import (
    CUSTOM_REFERENCE_GENOME,
    MANIFEST_FIELDS,
    TYPE_CASES,
    assert_has_type,
    assert_same_dataset,
    custom_reference_genome,
    decode_schema,
    field,
    parse_extension,
    read_manifest,
    read_parquet_file,
    required_by_index,
    row_files,
    to_python,
)
from ..helpers import resource


def write_columnar(ht: hl.Table, path: str, overwrite: bool = False) -> None:
    Env.backend().execute(ir.TableWrite(ht._tir, ir.TableColumnarWriter(path, overwrite)))


def keyed_table(n_rows: int = 10, n_partitions: int = 3) -> hl.Table:
    """A keyed table with primitive, struct and array fields. The fixtures in
    ``resources/columnar`` hold the same rows, and must change with it."""
    ht = hl.utils.range_table(n_rows, n_partitions=n_partitions)
    ht = ht.annotate(
        s=hl.str(ht.idx),
        f64=hl.float64(ht.idx) / 2,
        f32=hl.or_missing(ht.idx % 3 != 0, hl.float32(ht.idx)),
        i64=hl.int64(ht.idx) * 1_000_000_000_000,
        b=ht.idx % 2 == 0,
        ms=hl.missing(hl.tstr),
        arr=hl.range(ht.idx % 3).map(lambda i: hl.or_missing(i != 1, i)),
        st=hl.struct(x=ht.idx, y=hl.or_missing(ht.idx % 2 == 0, 'y')),
    )
    return ht.annotate_globals(g=5, gs='hello')


def test_manifest_is_well_formed():
    path = new_temp_file(extension='ht')
    write_columnar(keyed_table(), path)

    manifest = read_manifest(path)
    assert set(manifest) == MANIFEST_FIELDS
    assert manifest['format'] == 'hail-columnar'
    assert manifest['version'] == '0.1'
    assert manifest['kind'] == 'table'
    assert set(manifest['rows']) == {'schema', 'key', 'distinctly_keyed'}
    assert manifest['rows']['key'] == ['idx']
    assert manifest['rows']['distinctly_keyed'] is True
    assert set(manifest['globals']) == {'schema', 'file'}
    assert manifest['globals']['file'] == {'path': 'globals.parquet', 'container': 'parquet'}
    assert manifest['partitions'] == {'path': 'partitions.parquet', 'container': 'parquet'}
    assert manifest['internal_fields'] == {'row_id': '__row_id'}
    assert manifest['reference_genomes'] == []
    assert not hfs.exists(f'{path}/_SUCCESS')


def test_rows_equal_collected_values_and_row_ids_encode_position():
    ht = keyed_table()
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    row_id = read_manifest(path)['internal_fields']['row_id']
    rows = []
    for partition, file in row_files(path):
        part_rows = file.read().to_pylist()
        assert [r[row_id] for r in part_rows] == [
            (partition['partition_index'] << 40) | i for i in range(len(part_rows))
        ]
        rows.extend({k: v for k, v in r.items() if k != row_id} for r in part_rows)

    assert rows == [to_python(r) for r in ht.collect()]


KEYED_ROW_SCHEMA = pa.schema([
    field('__row_id', pa.int64(), 1, nullable=False),
    field('idx', pa.int32(), 2, nullable=False),
    field('s', pa.string(), 3, nullable=False),
    field('f64', pa.float64(), 4, nullable=False),
    field('f32', pa.float32(), 5),
    field('i64', pa.int64(), 6, nullable=False),
    field('b', pa.bool_(), 7, nullable=False),
    field('ms', pa.string(), 8),
    field('arr', pa.list_(field('element', pa.int32(), 10)), 9, nullable=False),
    field(
        'st', pa.struct([field('x', pa.int32(), 12, nullable=False), field('y', pa.string(), 13)]), 11, nullable=False
    ),
])


def test_row_files_have_the_manifest_schema():
    path = new_temp_file(extension='ht')
    write_columnar(keyed_table(), path)

    schema = decode_schema(read_manifest(path)['rows']['schema'])
    assert schema.equals(KEYED_ROW_SCHEMA, check_metadata=True), schema
    for _, file in row_files(path):
        assert file.schema_arrow.equals(schema, check_metadata=True)
        footer = file.metadata.metadata[b'ARROW:schema']
        assert pa.ipc.read_schema(pa.py_buffer(base64.b64decode(footer))).equals(schema, check_metadata=True)


KEYED_ROW_PARQUET_SCHEMA = """\
required group field_id=-1 schema {
  required int64 field_id=1 __row_id;
  required int32 field_id=2 idx;
  required binary field_id=3 s (String);
  required double field_id=4 f64;
  optional float field_id=5 f32;
  required int64 field_id=6 i64;
  required boolean field_id=7 b;
  optional binary field_id=8 ms (String);
  required group field_id=9 arr (List) {
    repeated group field_id=-1 list {
      optional int32 field_id=10 element;
    }
  }
  required group field_id=11 st {
    required int32 field_id=12 x;
    optional binary field_id=13 y (String);
  }
}"""


def check_parquet_requirements(file: pq.ParquetFile) -> None:
    """Checks the spec section 4.6 requirements that pyarrow exposes, for every row group."""
    metadata = file.metadata
    assert metadata.num_row_groups > 0
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        assert file.read_row_group(i).num_rows == row_group.num_rows
        for j in range(row_group.num_columns):
            column = row_group.column(j)
            assert column.compression == 'ZSTD', column.path_in_schema
            assert column.is_stats_set, column.path_in_schema
            assert column.has_column_index, column.path_in_schema
            assert column.has_offset_index, column.path_in_schema


def test_data_files_follow_the_parquet_requirements():
    path = new_temp_file(extension='ht')
    write_columnar(keyed_table(), path)

    for _, file in row_files(path):
        parquet_schema = str(file.schema).split('\n', 1)[1].strip()
        assert parquet_schema == KEYED_ROW_PARQUET_SCHEMA
        check_parquet_requirements(file)
    check_parquet_requirements(read_parquet_file(f'{path}/partitions.parquet'))
    check_parquet_requirements(read_parquet_file(f'{path}/globals.parquet'))


def key_tuple(bound: dict, key: list) -> tuple:
    return tuple(bound[k] for k in key)


def check_partitions_table(path: str) -> list:
    """Checks the partitions table against the row files, and returns its entries."""
    manifest = read_manifest(path)
    key = manifest['rows']['key']
    partitions = []
    for partition, file in row_files(path):
        rows = file.read().to_pylist()
        assert partition['row_count'] == len(rows)
        assert partition['row_file']['container'] == 'parquet'
        if not key:
            assert 'start' not in partition and 'end' not in partition
        elif rows:
            assert partition['start'] == {k: rows[0][k] for k in key}
            assert partition['end'] == {k: rows[-1][k] for k in key}
            sort_keys = [key_tuple(r, key) for r in rows]
            assert sort_keys == sorted(sort_keys)
        else:
            assert partition['start'] is None and partition['end'] is None
        partitions.append(partition)

    assert [p['partition_index'] for p in partitions] == list(range(len(partitions)))
    if key:
        non_empty = [p for p in partitions if p['row_count'] > 0]
        for before, after in pairwise(non_empty):
            assert key_tuple(before['end'], key) < key_tuple(after['start'], key)
    return partitions


def test_partitions_table_has_tight_increasing_bounds():
    ht = hl.utils.range_table(9, n_partitions=3)
    # Not a key-range filter, which would drop the emptied partition.
    ht = ht.filter(~hl.set({'3', '4', '5'}).contains(hl.str(ht.idx)))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    partitions = check_partitions_table(path)
    assert [p['row_count'] for p in partitions] == [3, 0, 3]

    schema = read_parquet_file(f'{path}/partitions.parquet').schema_arrow
    expected = pa.schema([
        field('partition_index', pa.int32(), 1, nullable=False),
        field('start', pa.struct([field('idx', pa.int32(), 3, nullable=False)]), 2),
        field('end', pa.struct([field('idx', pa.int32(), 5, nullable=False)]), 4),
        field('row_count', pa.int64(), 6, nullable=False),
        field(
            'row_file',
            pa.struct([
                field('path', pa.string(), 8, nullable=False),
                field('container', pa.string(), 9, nullable=False),
            ]),
            7,
            nullable=False,
        ),
    ])
    assert schema.equals(expected, check_metadata=True), schema


def test_globals_file_holds_the_globals():
    path = new_temp_file(extension='ht')
    write_columnar(keyed_table(), path)

    schema = decode_schema(read_manifest(path)['globals']['schema'])
    expected = pa.schema([field('g', pa.int32(), 1, nullable=False), field('gs', pa.string(), 2, nullable=False)])
    assert schema.equals(expected, check_metadata=True), schema
    file = read_parquet_file(f'{path}/globals.parquet')
    assert file.schema_arrow.equals(schema, check_metadata=True)
    assert file.read().to_pylist() == [{'g': 5, 'gs': 'hello'}]


def test_empty_globals_have_a_schema_and_no_file():
    path = new_temp_file(extension='ht')
    write_columnar(hl.utils.range_table(5, n_partitions=2), path)

    manifest = read_manifest(path)
    assert manifest['globals']['file'] is None
    assert decode_schema(manifest['globals']['schema']).equals(pa.schema([]), check_metadata=True)
    assert not hfs.exists(f'{path}/globals.parquet')


def test_unkeyed_table():
    ht = keyed_table().key_by()
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    manifest = read_manifest(path)
    assert manifest['rows']['key'] == []
    assert manifest['rows']['distinctly_keyed'] is False
    check_partitions_table(path)
    schema = read_parquet_file(f'{path}/partitions.parquet').schema_arrow
    assert schema.names == ['partition_index', 'row_count', 'row_file']

    row_id = manifest['internal_fields']['row_id']
    rows = [{k: v for k, v in r.items() if k != row_id} for _, f in row_files(path) for r in f.read().to_pylist()]
    assert rows == [to_python(r) for r in ht.collect()]


def test_rows_sharing_a_key_are_never_split_across_partitions():
    ht = hl.utils.range_table(12, n_partitions=4)
    ht = ht.annotate(a=ht.idx // 6).key_by('a', 'idx')
    # On a lowered backend, keying by a prefix coarsens the partitioner without a shuffle, so key
    # a=0 spans partitions 0 and 1, and a=1 spans 2 and 3. The Spark backend strictifies the
    # partitioner itself before the write.
    ht = ht.key_by('a')
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    partitions = check_partitions_table(path)
    assert [(p['start'], p['end']) for p in partitions] == [({'a': 0}, {'a': 0}), ({'a': 1}, {'a': 1})]
    manifest = read_manifest(path)
    assert manifest['rows']['distinctly_keyed'] is False
    row_id = manifest['internal_fields']['row_id']
    rows = [{k: v for k, v in r.items() if k != row_id} for _, f in row_files(path) for r in f.read().to_pylist()]
    assert rows == [to_python(r) for r in ht.collect()]


def test_a_failed_write_is_not_committed():
    ht = hl.utils.range_table(10, n_partitions=2)
    ht = ht.annotate(x=hl.case().when(ht.idx < 7, ht.idx).or_error('boom'))
    path = new_temp_file(extension='ht')
    with pytest.raises((FatalError, HailUserError), match='boom'):
        write_columnar(ht, path)

    assert not hfs.exists(f'{path}/manifest.json')
    assert not hfs.exists(f'{path}/_SUCCESS')
    assert not is_committed(path)


def test_row_id_field_name_does_not_collide_with_row_fields():
    ht = hl.utils.range_table(3, n_partitions=1)
    ht = ht.annotate(__row_id=ht.idx * 10, __row_id1=ht.idx * 100)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    assert read_manifest(path)['internal_fields'] == {'row_id': '__row_id2'}
    ((_, file),) = row_files(path)
    assert file.read().to_pylist() == [
        {'__row_id2': i, 'idx': i, '__row_id': i * 10, '__row_id1': i * 100} for i in range(3)
    ]


def write_typed_fields(values: list, typ: hl.HailType) -> tuple:
    """Writes a table whose row i has a required field ``x`` holding ``values[i]``, and a nullable field
    ``x_opt`` holding the same value in even rows and missing in odd rows. Returns the fields' Arrow
    schemas and their values as pyarrow reads them."""
    ht = hl.utils.range_table(len(values), n_partitions=2)
    ht = ht.annotate(x=required_by_index(ht.idx, values, typ))
    ht = ht.annotate(x_opt=hl.or_missing(ht.idx % 2 == 0, ht.x))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    schema = decode_schema(read_manifest(path)['rows']['schema'])
    rows = [r for _, f in row_files(path) for r in f.read().to_pylist()]
    return schema.field('x'), schema.field('x_opt'), [r['x'] for r in rows], [r['x_opt'] for r in rows]


@pytest.mark.parametrize('case', TYPE_CASES.keys())
def test_types_have_the_specified_extension_and_encoding(case):
    c = TYPE_CASES[case]()
    x, x_opt, stored, stored_opt = write_typed_fields(c.values, c.typ)

    for f in (x, x_opt):
        assert_has_type(f, c)
    assert not x.nullable
    assert x_opt.nullable
    assert stored == c.stored
    assert stored_opt == [v if i % 2 == 0 else None for i, v in enumerate(c.stored)]


def test_manifest_defines_only_non_builtin_reference_genomes():
    rg = custom_reference_genome()
    ht = hl.utils.range_table(2)
    ht = ht.annotate(l=hl.locus('1', ht.idx + 1, reference_genome='GRCh37'))
    ht = ht.annotate_globals(
        ivs=[hl.interval(hl.locus('chr1', 1, reference_genome=rg), hl.locus('chr1', 5, reference_genome=rg))]
    )
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    assert read_manifest(path)['reference_genomes'] == [CUSTOM_REFERENCE_GENOME]


def column_index(file: pq.ParquetFile, path: str) -> int:
    """The index of the Parquet leaf column at ``path``."""
    return next(i for i in range(file.metadata.num_columns) if file.schema.column(i).path == path)


def test_locus_keys_are_ordered_by_the_reference_genome():
    rg = custom_reference_genome()
    # Lexicographic order, which differs from the genome's.
    contigs = ['chr1', 'chr10', 'chr2', 'chrX']
    ht = hl.utils.range_table(8, n_partitions=3)
    ht = ht.key_by(locus=hl.locus(hl.literal(contigs)[ht.idx % 4], ht.idx // 4 + 1, reference_genome=rg))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    # (contig_index << 32) | position, in the genome's contig order chr2, chr10, chr1, chrX.
    expected = [(c << 32) | p for c in range(4) for p in (1, 2)]
    partitions = check_partitions_table(path)
    assert [r['locus'] for _, f in row_files(path) for r in f.read().to_pylist()] == expected
    for _, file in row_files(path):
        locus = column_index(file, 'locus')
        for i in range(file.metadata.num_row_groups):
            loci = file.read_row_group(i, columns=['locus']).column('locus').to_pylist()
            statistics = file.metadata.row_group(i).column(locus).statistics
            assert (statistics.min, statistics.max) == (min(loci), max(loci))

    # The bounds have the key's Arrow type, extension type included (spec section 4.1).
    locus_extension = ('hail.locus', {'reference_genome': rg.name, 'contigs': rg.contigs})
    bounds = read_parquet_file(f'{path}/partitions.parquet').schema_arrow
    for bound in ('start', 'end'):
        assert parse_extension(bounds.field(bound).type.field('locus')) == locus_extension
    non_empty = [p for p in partitions if p['row_count'] > 0]
    assert non_empty[0]['start'] == {'locus': expected[0]}
    assert non_empty[-1]['end'] == {'locus': expected[-1]}


def test_ndarrays_are_stored_row_major():
    ht = hl.utils.range_table(1)
    ht = ht.annotate(x=hl.nd.array([[1, 2, 3], [4, 5, 6]]).T)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    ((_, file),) = row_files(path)
    assert file.read().to_pylist()[0]['x'] == {'data': [1, 4, 2, 5, 3, 6], 'shape': [3, 2]}


def test_an_ndarray_dimension_longer_than_int32_fails_the_write():
    ht = hl.utils.range_table(1)
    # No elements, so nothing large is allocated.
    ht = ht.annotate(x=hl.nd.zeros((2**31, 0)))
    path = new_temp_file(extension='ht')
    with pytest.raises(FatalError, match="ndarray in field 'x' has a dimension longer than 2147483647"):
        write_columnar(ht, path)
    assert not is_committed(path)


def test_a_missing_dict_key_fails_the_write():
    ht = hl.utils.range_table(4, n_partitions=2)
    ht = ht.annotate(s=hl.struct(d=hl.dict([(hl.or_missing(ht.idx != 2, 'k'), ht.idx)])))
    path = new_temp_file(extension='ht')
    with pytest.raises(FatalError, match="field 's.d' has a missing dict key.*Filter the missing dict key out"):
        write_columnar(ht, path)
    assert not is_committed(path)


def test_a_row_field_named_like_the_row_uid_fails_the_write():
    ht = hl.utils.range_table(1)
    ht = ht.annotate(**{'__row_uid': ht.idx})
    path = new_temp_file(extension='ht')
    with pytest.raises(FatalError, match="row field '__row_uid' can't be stored"):
        write_columnar(ht, path)
    assert not is_committed(path)


def test_overwrite():
    path = new_temp_file(extension='ht')
    write_columnar(hl.utils.range_table(3), path)
    with pytest.raises(FatalError, match='already exists'):
        write_columnar(hl.utils.range_table(5), path)

    write_columnar(hl.utils.range_table(5), path, overwrite=True)
    assert sum(p['row_count'] for p in check_partitions_table(path)) == 5


def all_types_values(rg: hl.ReferenceGenome) -> list:
    """For each type the columnar format can store, except binary, which Hail's Python API has no type
    for, its name, Hail type, and the value of row ``i`` of the ``all_types`` fixture, for ``i`` in 0
    to 3. ``generate.py`` holds the same values, encoded as
    spec section 5 says, and must change with these."""
    contigs = rg.contigs
    calls = [hl.Call([0, 1]), hl.Call([1, 1]), hl.Call([0, 1], phased=True), hl.Call([1])]
    return [
        ('bool', hl.tbool, lambda i: i % 2 == 0),
        ('int32', hl.tint32, lambda i: i - 1),
        ('int64', hl.tint64, lambda i: i * 1_000_000_000_000 - 1),
        ('float32', hl.tfloat32, lambda i: i / 4),
        ('float64', hl.tfloat64, lambda i: i * 1.5),
        ('str', hl.tstr, lambda i: f'v{i}'),
        ('struct', hl.tstruct(a=hl.tint32, b=hl.tstr), lambda i: hl.Struct(a=i, b=str(i))),
        ('empty_struct', hl.tstruct(), lambda i: hl.Struct()),
        ('tuple', hl.ttuple(hl.tint32, hl.tstr), lambda i: (i, str(i))),
        ('empty_tuple', hl.ttuple(), lambda i: ()),
        ('array', hl.tarray(hl.tint32), lambda i: list(range(i))),
        ('set', hl.tset(hl.tint32), lambda i: {3 - j for j in range(i)}),
        ('dict', hl.tdict(hl.tstr, hl.tint32), lambda i: {str(3 - j): None if j == 1 else j for j in range(i)}),
        ('locus', hl.tlocus(rg), lambda i: hl.Locus(contigs[i], i + 1, rg)),
        ('builtin_locus', hl.tlocus('GRCh37'), lambda i: hl.Locus(str(i + 1), 100 + i, 'GRCh37')),
        (
            'interval',
            hl.tinterval(hl.tlocus(rg)),
            lambda i: hl.Interval(hl.Locus(contigs[i], 1, rg), hl.Locus(contigs[i], 10 + i, rg), True, i % 2 == 0),
        ),
        ('call', hl.tcall, lambda i: calls[i]),
        ('ndarray', hl.tndarray(hl.tfloat64, 2), lambda i: np.arange(i * 2, dtype=np.float64).reshape((i, 2))),
    ]


def all_types_table() -> hl.Table:
    """A keyed table of 4 rows in 2 row partitions with, for every type the columnar format can store, a
    required field, a nullable field missing in odd rows, and the same value nested in a struct, alone
    and in arrays with required and nullable elements."""
    ht = hl.utils.range_table(4, n_partitions=2)
    fields = {}
    for name, typ, value in all_types_values(custom_reference_genome()):
        x = required_by_index(ht.idx, [value(i) for i in range(4)], typ)
        x_opt = hl.or_missing(ht.idx % 2 == 0, x)
        fields[name] = x
        fields[f'{name}_opt'] = x_opt
        fields[f'{name}_nested'] = hl.struct(
            req=x, opt=x_opt, req_arr=hl.array([x]), opt_arr=hl.array([x, hl.missing(typ)])
        )
    return ht.annotate(**fields).annotate_globals(g=5, gs='hello')


def no_globals_table() -> hl.Table:
    ht = hl.utils.range_table(6, n_partitions=2)
    return ht.annotate(s=hl.str(ht.idx))


FIXTURES = [
    ('keyed', lambda: keyed_table(6, 2)),
    ('unkeyed', lambda: keyed_table(6, 2).key_by()),
    ('no_globals', no_globals_table),
    ('all_types', all_types_table),
]


@pytest.mark.parametrize('fixture, table', FIXTURES)
def test_output_matches_fixture(fixture, table):
    expected = resource(f'columnar/0.1/table/{fixture}')
    check_partitions_table(expected)
    path = new_temp_file(extension='ht')
    write_columnar(table(), path)

    assert_same_dataset(path, expected)


def read_columnar(path: str) -> hl.Table:
    return hl.Table(ir.TableRead(ir.TableColumnarReader(path)))


def assert_reads_as(path: str, expected: hl.Table) -> None:
    """The columnar table at ``path`` reads as ``expected``: the same key, types, values and row order."""
    actual = read_columnar(path)
    assert list(actual.key) == list(expected.key)
    assert actual._same(expected)
    assert actual.aggregate(hl.agg.collect(actual.idx)) == expected.aggregate(hl.agg.collect(expected.idx))


@pytest.mark.parametrize('fixture, table', FIXTURES)
def test_fixture_reads_as_the_table_it_holds(fixture, table):
    assert_reads_as(resource(f'columnar/0.1/table/{fixture}'), table())


@pytest.mark.parametrize('case', TYPE_CASES.keys())
def test_round_trip_preserves_values(case):
    c = TYPE_CASES[case]()
    ht = hl.utils.range_table(len(c.values), n_partitions=2)
    ht = ht.annotate(x=required_by_index(ht.idx, c.values, c.typ))
    ht = ht.annotate(x_opt=hl.or_missing(ht.idx % 2 == 0, ht.x), x_arr=hl.array([ht.x, hl.missing(c.typ)]))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    assert_reads_as(path, ht)


def test_round_trip_of_every_type_preserves_values():
    ht = all_types_table()
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    assert_reads_as(path, ht)


def local_path(path: str) -> str:
    """``path``, a file in a local temporary directory, as a path Python's ``open`` takes."""
    return path.removeprefix('file://')


def drop_checksum(file_path: str) -> None:
    """Removes the checksum file Hadoop's local file system keeps beside ``file_path``, which would fail
    reads of a file changed outside Hadoop."""
    directory, name = local_path(file_path).rsplit('/', 1)
    crc = f'{directory}/.{name}.crc'
    if hfs.exists(crc):
        hfs.remove(crc)


def row_file_paths(path: str) -> list:
    return [f'{path}/{p["row_file"]["path"]}' for p, _ in row_files(path)]


def corrupt_column_chunks_except(file_path: str, keep: set) -> None:
    """Overwrites with zeros every column chunk of the Parquet file at ``file_path`` whose leaf column
    path is not in ``keep``. Reading a zeroed chunk fails."""
    with hfs.open(file_path, 'rb') as f:
        data = bytearray(f.read())
    metadata = pq.ParquetFile(io.BytesIO(data)).metadata
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        for j in range(row_group.num_columns):
            column = row_group.column(j)
            if column.path_in_schema in keep:
                continue
            start = column.dictionary_page_offset if column.has_dictionary_page else column.data_page_offset
            data[start : start + column.total_compressed_size] = bytes(column.total_compressed_size)
    with hfs.open(file_path, 'wb') as f:
        f.write(data)
    drop_checksum(file_path)


def test_projection_reads_only_the_requested_fields_column_chunks():
    ht = keyed_table(6, 2)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    for file_path in row_file_paths(path):
        corrupt_column_chunks_except(file_path, {'idx', 's', 'st.x'})

    def project(t: hl.Table) -> hl.Table:
        return t.select('s', x=t.st.x)

    assert project(read_columnar(path)).collect() == project(ht).collect()
    # Reading any other field reads a zeroed chunk.
    with pytest.raises(FatalError):
        read_columnar(path).select('f64').collect()


def test_files_the_manifest_does_not_reference_are_ignored():
    ht = keyed_table(6, 2)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    # A failed task's output: a copy of a row file under another name, which would duplicate its rows.
    first_row_file = row_file_paths(path)[0]
    hfs.copy(first_row_file, first_row_file.replace('-0000-4', '-1111-4', 1).replace('.parquet', '-orphan.parquet'))
    with hfs.open(f'{path}/rows/garbage.parquet', 'wb') as f:
        f.write(b'not parquet')

    assert_reads_as(path, ht)


def test_key_and_partitioning_come_from_the_manifest():
    path = resource('columnar/0.1/table/keyed')
    ht = read_columnar(path)

    assert list(ht.key) == ['idx']
    assert ht.n_partitions() == 2
    assert ht.aggregate(hl.agg.collect(ht.idx)) == list(range(6))


def test_row_partitions_with_no_rows_are_dropped():
    # The predicate is on a string so that it isn't pushed into the range table's partitioning.
    ht = hl.utils.range_table(10, n_partitions=5)
    ht = ht.filter(hl.literal({'0', '1', '8', '9'}).contains(hl.str(ht.idx)))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    assert sum(p['row_count'] == 0 for p in check_partitions_table(path)) == 3

    actual = read_columnar(path)
    assert actual.n_partitions() == 2
    assert actual._same(ht)


def unique_reference_genome_name() -> str:
    return f'ColumnarRead{secrets.token_hex(4)}'


def write_with_reference_genome(name: str) -> str:
    """Writes a table with a locus field on a new reference genome ``name``, which is then removed from the
    session. Returns the table's path."""
    rg = hl.ReferenceGenome(name, ['b', 'a'], {'b': 100, 'a': 50})
    ht = hl.utils.range_table(2)
    ht = ht.annotate(l=hl.locus(hl.literal(['a', 'b'])[ht.idx], ht.idx + 1, reference_genome=rg))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    Env.backend().remove_reference(name)
    return path


def test_reading_defines_the_tables_reference_genomes():
    name = unique_reference_genome_name()
    path = write_with_reference_genome(name)

    ht = read_columnar(path)
    assert hl.get_reference(name).contigs == ['b', 'a']
    assert hl.get_reference(name).lengths == {'b': 100, 'a': 50}
    assert ht.aggregate(hl.agg.collect((ht.l.contig, ht.l.position))) == [('a', 1), ('b', 2)]
    Env.backend().remove_reference(name)


def test_a_reference_genome_that_differs_from_the_sessions_fails_the_read():
    name = unique_reference_genome_name()
    path = write_with_reference_genome(name)
    hl.ReferenceGenome(name, ['a'], {'a': 50})
    try:
        with pytest.raises(FatalError, match='a different reference with that name already exists'):
            read_columnar(path)
    finally:
        Env.backend().remove_reference(name)


def rewrite_row_file(path: str, f) -> None:
    """Rewrites the table's first row file with pyarrow, replacing its table ``t`` with ``f(t)``. The file
    keeps its Arrow schema."""
    file_path = row_file_paths(path)[0]
    table = read_parquet_file(file_path).read()
    with hfs.open(file_path, 'wb') as out:
        pq.write_table(f(table), out, store_schema=True)
    drop_checksum(file_path)


def write_row_file_with_manifest_schema(path: str, table: pa.Table) -> None:
    """Replaces the table's first row file with ``table``, with the manifest's Arrow schema in its footer
    in place of ``table``'s, so the file passes the reader's Arrow schema check whatever it holds."""
    file_path = row_file_paths(path)[0]
    with hfs.open(file_path, 'wb') as out, pq.ParquetWriter(out, table.schema, store_schema=False) as writer:
        writer.write_table(table)
        writer.add_key_value_metadata({'ARROW:schema': read_manifest(path)['rows']['schema']})
    drop_checksum(file_path)


def rewrite_ndarray_row_file(path: str, n_dims: int, data: list, shape: Optional[list]) -> None:
    """Replaces the first row file of a one-row table whose ``x`` is an ``n_dims``-dimensional ndarray of
    float64 with one whose ``x`` has ``data`` and ``shape``."""
    # pyarrow writes its own storage type for a canonical tensor, so the file is written from its storage
    # type, with the manifest's Arrow schema in its footer.
    stored = read_parquet_file(row_file_paths(path)[0]).schema_arrow
    x = pa.struct([
        pa.field('data', pa.list_(pa.field('element', pa.float64(), False)), False),
        pa.field('shape', pa.list_(pa.field('element', pa.int32(), False), n_dims)),
    ])
    schema = pa.schema([stored.field('__row_id'), stored.field('idx'), pa.field('x', x, stored.field('x').nullable)])
    table = pa.table({'__row_id': [0], 'idx': [0], 'x': [{'data': data, 'shape': shape}]}, schema=schema)
    write_row_file_with_manifest_schema(path, table)


def test_row_groups_with_no_rows_are_read():
    ht = keyed_table(6, 2)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    file_path = row_file_paths(path)[0]
    table = read_parquet_file(file_path).read()
    with hfs.open(file_path, 'wb') as out, pq.ParquetWriter(out, table.schema, store_schema=True) as writer:
        writer.write_table(table.slice(0, 0))
        writer.write_table(table)
        writer.write_table(table.slice(0, 0))
    drop_checksum(file_path)
    metadata = read_parquet_file(file_path).metadata
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [0, 3, 0]

    assert_reads_as(path, ht)


def test_a_row_file_whose_schema_differs_from_the_manifest_fails_the_read():
    ht = keyed_table(6, 2)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    rewrite_row_file(path, lambda t: t.drop_columns(['ms']))

    with pytest.raises(FatalError, match='differs from its manifest'):
        read_columnar(path).collect()


def test_a_row_file_missing_a_parquet_column_fails_the_read():
    ht = keyed_table(6, 2)
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    table = read_parquet_file(row_file_paths(path)[0]).read()
    write_row_file_with_manifest_schema(
        path, table.rename_columns(['renamed' if name == 's' else name for name in table.column_names])
    )

    with pytest.raises(FatalError, match=r"has no Parquet column s\b"):
        read_columnar(path).collect()


def test_an_empty_struct_stored_as_false_fails_the_read():
    ht = hl.utils.range_table(2, n_partitions=1)
    ht = ht.annotate(x=hl.struct())
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    rewrite_row_file(
        path, lambda t: t.set_column(t.schema.get_field_index('x'), t.schema.field('x'), pa.array([True, False]))
    )

    with pytest.raises(FatalError, match='stored as false'):
        read_columnar(path).collect()


def test_a_present_ndarray_with_a_null_shape_fails_the_read():
    ht = hl.utils.range_table(1, n_partitions=1)
    ht = ht.annotate(x=hl.nd.array([1.0, 2.0]))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    rewrite_ndarray_row_file(path, 1, [1.0, 2.0], None)

    with pytest.raises(FatalError, match='null shape'):
        read_columnar(path).collect()


def test_an_ndarray_with_a_negative_dimension_fails_the_read():
    ht = hl.utils.range_table(1, n_partitions=1)
    ht = ht.annotate(x=hl.nd.array([[1.0], [2.0]]))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    # The product of the dimensions is the data's length.
    rewrite_ndarray_row_file(path, 2, [1.0, 2.0], [-1, -2])

    with pytest.raises(FatalError, match='negative dimension'):
        read_columnar(path).collect()


def test_an_ndarray_whose_shape_overflows_fails_the_read():
    ht = hl.utils.range_table(1, n_partitions=1)
    ht = ht.annotate(x=hl.nd.array([[[1.0]]]))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)
    # 2^22 * 2^22 * 2^20 wraps to 0 in a 64-bit product.
    rewrite_ndarray_row_file(path, 3, [], [1 << 22, 1 << 22, 1 << 20])

    with pytest.raises(FatalError, match='does not match its shape'):
        read_columnar(path).collect()


def test_round_trip_of_a_table_keyed_by_locus():
    rg = custom_reference_genome()
    contigs = ['chr1', 'chr10', 'chr2', 'chrX']
    ht = hl.utils.range_table(8, n_partitions=3)
    ht = ht.key_by(locus=hl.locus(hl.literal(contigs)[ht.idx % 4], ht.idx // 4 + 1, reference_genome=rg))
    path = new_temp_file(extension='ht')
    write_columnar(ht, path)

    actual = read_columnar(path)
    assert actual._same(ht)
    # In the genome's contig order, chr2, chr10, chr1, chrX, not lexicographic order.
    assert actual.aggregate(hl.agg.collect(actual.locus.contig)) == [
        c for c in ['chr2', 'chr10', 'chr1', 'chrX'] for _ in (1, 2)
    ]


def test_reading_no_field_of_a_struct_keeps_whether_it_is_missing_and_its_length():
    def project(t: hl.Table) -> hl.Table:
        n = t.struct_nested
        return t.select(
            defined=hl.is_defined(t.struct_opt),
            n_opt_arr=hl.len(n.opt_arr),
            defined_elements=n.opt_arr.map(hl.is_defined),
        )

    actual = project(read_columnar(resource('columnar/0.1/table/all_types')))
    expected = project(all_types_table())
    assert actual.collect() == expected.collect()

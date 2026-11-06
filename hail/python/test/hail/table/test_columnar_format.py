"""Conformance tests for tables written in the columnar format.

These read Hail's output with pyarrow, independently of Hail, and check it against the columnar
format specification (``dev-docs/hail-query/columnar-format-spec.md``).
"""

import base64
import io
import json
import struct
from itertools import pairwise

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hail as hl
import hailtop.fs as hfs
from hail import ir
from hail.utils import FatalError, HailUserError, new_temp_file
from hail.utils.java import Env
from hail.utils.misc import is_committed

from ..helpers import resource

MANIFEST_FIELDS = {
    'format',
    'version',
    'kind',
    'rows',
    'globals',
    'partitions',
    'internal_fields',
    'reference_genomes',
}


def write_columnar(ht: hl.Table, path: str, overwrite: bool = False) -> None:
    Env.backend().execute(ir.TableWrite(ht._tir, ir.TableColumnarWriter(path, overwrite)))


def read_manifest(path: str) -> dict:
    with hfs.open(f'{path}/manifest.json') as f:
        return json.load(f)


def read_parquet_file(path: str) -> pq.ParquetFile:
    with hfs.open(path, 'rb') as f:
        return pq.ParquetFile(io.BytesIO(f.read()))


def keyed_table(n_rows: int = 10, n_partitions: int = 3) -> hl.Table:
    """A keyed table with every type the writer supports so far. The fixtures in
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


def to_python(value):
    """A collected Hail value as the plain Python value pyarrow's ``to_pylist`` gives."""
    if isinstance(value, hl.Struct):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


def row_files(path: str) -> list:
    """Each row partition's partitions-table entry, with its row file."""
    partitions = read_parquet_file(f'{path}/partitions.parquet').read().to_pylist()
    return [(p, read_parquet_file(f'{path}/{p["row_file"]["path"]}')) for p in partitions]


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


def field(name: str, typ: pa.DataType, field_id: int, nullable: bool = True) -> pa.Field:
    return pa.field(name, typ, nullable, {b'PARQUET:field_id': str(field_id).encode()})


def decode_schema(encoded: str) -> pa.Schema:
    """Decodes a manifest ``schema``, checking that it is an encapsulated IPC Schema message."""
    message = base64.b64decode(encoded, validate=True)
    continuation, length = struct.unpack('<Ii', message[:8])
    assert continuation == 0xFFFFFFFF
    assert len(message) == 8 + length
    assert len(message) % 8 == 0
    return pa.ipc.read_schema(pa.py_buffer(message))


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


@pytest.mark.parametrize(
    'annotate, message',
    [
        (lambda ht: ht.annotate(l=hl.locus('1', ht.idx + 1)), "row field 'l' has type locus<GRCh37>"),
        (lambda ht: ht.annotate(s=hl.struct(e=hl.struct())), "row field 's.e' is an empty struct"),
        (lambda ht: ht.annotate_globals(d=hl.dict({1: 2})), "global field 'd' has type dict<int32, int32>"),
    ],
)
def test_types_the_writer_does_not_support_yet_are_rejected(annotate, message):
    path = new_temp_file(extension='ht')
    with pytest.raises(FatalError, match=message):
        write_columnar(annotate(hl.utils.range_table(3)), path)
    assert not is_committed(path)


def test_overwrite():
    path = new_temp_file(extension='ht')
    write_columnar(hl.utils.range_table(3), path)
    with pytest.raises(FatalError, match='already exists'):
        write_columnar(hl.utils.range_table(5), path)

    write_columnar(hl.utils.range_table(5), path, overwrite=True)
    assert sum(p['row_count'] for p in check_partitions_table(path)) == 5


def without_schemas(manifest: dict) -> dict:
    return {k: {**v, 'schema': None} if isinstance(v, dict) and 'schema' in v else v for k, v in manifest.items()}


def assert_same_table(actual: str, expected: str) -> None:
    """Compares two columnar tables by value. Row file paths may differ."""
    actual_manifest = read_manifest(actual)
    expected_manifest = read_manifest(expected)
    assert without_schemas(actual_manifest) == without_schemas(expected_manifest)
    for component in ('rows', 'globals'):
        actual_schema = decode_schema(actual_manifest[component]['schema'])
        assert actual_schema.equals(decode_schema(expected_manifest[component]['schema']), check_metadata=True)

    actual_partitions = row_files(actual)
    expected_partitions = row_files(expected)
    assert len(actual_partitions) == len(expected_partitions)
    for (actual_partition, actual_file), (expected_partition, expected_file) in zip(
        actual_partitions, expected_partitions
    ):
        del actual_partition['row_file']['path'], expected_partition['row_file']['path']
        assert actual_partition == expected_partition
        assert actual_file.schema_arrow.equals(expected_file.schema_arrow, check_metadata=True)
        assert actual_file.read().to_pylist() == expected_file.read().to_pylist()

    partitions_schema = read_parquet_file(f'{actual}/partitions.parquet').schema_arrow
    assert partitions_schema.equals(
        read_parquet_file(f'{expected}/partitions.parquet').schema_arrow, check_metadata=True
    )

    if expected_manifest['globals']['file'] is not None:
        actual_globals = read_parquet_file(f'{actual}/globals.parquet')
        expected_globals = read_parquet_file(f'{expected}/globals.parquet')
        assert actual_globals.schema_arrow.equals(expected_globals.schema_arrow, check_metadata=True)
        assert actual_globals.read().to_pylist() == expected_globals.read().to_pylist()


def no_globals_table() -> hl.Table:
    ht = hl.utils.range_table(6, n_partitions=2)
    return ht.annotate(s=hl.str(ht.idx))


@pytest.mark.parametrize(
    'fixture, table',
    [
        ('keyed', lambda: keyed_table(6, 2)),
        ('unkeyed', lambda: keyed_table(6, 2).key_by()),
        ('no_globals', no_globals_table),
    ],
)
def test_output_matches_fixture(fixture, table):
    expected = resource(f'columnar/0.1/table/{fixture}')
    check_partitions_table(expected)
    path = new_temp_file(extension='ht')
    write_columnar(table(), path)

    assert_same_table(path, expected)

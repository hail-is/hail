"""Conformance tests for matrix tables written in the columnar format.

These read Hail's output with pyarrow, independently of Hail, and check it against the columnar
format specification (``dev-docs/hail-query/columnar-format-spec.md``).
"""

import secrets

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import hail as hl
import hailtop.fs as hfs
from hail import ir
from hail.utils import FatalError, new_temp_file
from hail.utils.java import Env
from hail.utils.misc import is_committed

from ..columnar_helpers import (
    CUSTOM_REFERENCE_GENOME,
    MANIFEST_FIELDS,
    TYPE_CASES,
    assert_has_type,
    assert_same_dataset,
    corrupt_column_chunks_except,
    custom_reference_genome,
    data_files,
    decode_schema,
    drop_checksum,
    field,
    read_manifest,
    read_parquet_file,
    required_by_index,
    to_python,
)
from ..helpers import resource


def write_columnar(
    mt: hl.MatrixTable,
    path: str,
    overwrite: bool = False,
    leaf_cap: int = ir.MatrixColumnarWriter.LEAF_CAP,
    byte_cap: int = ir.MatrixColumnarWriter.BYTE_CAP,
) -> None:
    Env.backend().execute(ir.MatrixWrite(mt._mir, ir.MatrixColumnarWriter(path, overwrite, leaf_cap, byte_cap)))


def matrix_table(n_rows: int = 6, n_cols: int = 4, n_partitions: int = 2) -> hl.MatrixTable:
    """A matrix table with row, column, entry and global fields, missing entries, and present entries
    with a missing field. The ``missing_entries`` fixture in ``resources/columnar`` holds the default
    matrix table, and must change with it."""
    mt = hl.utils.range_matrix_table(n_rows, n_cols, n_partitions=n_partitions)
    mt = mt.annotate_rows(r=hl.str(mt.row_idx))
    mt = mt.annotate_cols(s=hl.str(mt.col_idx))
    mt = mt.annotate_entries(
        x=mt.row_idx * 10 + mt.col_idx,
        y=hl.or_missing((mt.row_idx + mt.col_idx) % 3 != 0, hl.str(mt.col_idx)),
    )
    mt = mt.filter_entries((mt.row_idx + mt.col_idx) % 4 != 1)
    return mt.annotate_globals(g=5)


def test_manifest_describes_a_matrix_table():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(n_cols=4), path)

    manifest = read_manifest(path)
    assert set(manifest) == MANIFEST_FIELDS | {'cols', 'entries'}
    assert manifest['kind'] == 'matrix_table'
    assert manifest['rows']['key'] == ['row_idx']
    assert set(manifest['cols']) == {'schema', 'key', 'count', 'file'}
    assert manifest['cols']['key'] == ['col_idx']
    assert manifest['cols']['count'] == 4
    assert manifest['cols']['file'] == {'path': 'cols.parquet', 'container': 'parquet'}
    assert set(manifest['entries']) == {'schema', 'layout', 'column_blocks'}
    assert manifest['entries']['layout'] == 'dense'
    assert manifest['entries']['column_blocks'] == [{'start': 0, 'end': 4}]
    assert manifest['internal_fields'] == {
        'row_id': '__row_id',
        'col_idx': '__col_idx',
        'entry_valid': '__entry_valid',
    }
    assert decode_schema(manifest['globals']['schema']).names == ['g']


ENTRY_SCHEMA = pa.schema([
    field('__row_id', pa.int64(), 1, nullable=False),
    field('__entry_valid', pa.list_(field('element', pa.bool_(), 3, nullable=False)), 2, nullable=False),
    field('x', pa.list_(field('element', pa.int32(), 5)), 4, nullable=False),
    field('y', pa.list_(field('element', pa.string(), 7)), 6, nullable=False),
])


def test_entry_files_have_the_manifest_schema():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)

    schema = decode_schema(read_manifest(path)['entries']['schema'])
    assert schema.equals(ENTRY_SCHEMA, check_metadata=True), schema
    for _, _, entry_files in data_files(path):
        assert len(entry_files) == 1
        assert entry_files[0].schema_arrow.equals(schema, check_metadata=True)


def test_entry_files_hold_each_rows_entries_by_column():
    n_rows, n_cols = 6, 4
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(n_rows, n_cols), path)

    row_id = read_manifest(path)['internal_fields']['row_id']
    entry_valid = read_manifest(path)['internal_fields']['entry_valid']
    entries = []
    for _, row_file, (entry_file,) in data_files(path):
        rows = row_file.read().to_pylist()
        part_entries = entry_file.read().to_pylist()
        assert [e[row_id] for e in part_entries] == [r[row_id] for r in rows]
        entries.extend(part_entries)

    assert len(entries) == n_rows
    for i, e in enumerate(entries):
        # matrix_table's definition: entry (i, j) is missing when (i + j) % 4 == 1, and its field y
        # is missing when (i + j) % 3 == 0.
        valid = [(i + j) % 4 != 1 for j in range(n_cols)]
        assert e[entry_valid] == valid
        assert e['x'] == [i * 10 + j if valid[j] else None for j in range(n_cols)]
        assert e['y'] == [str(j) if valid[j] and (i + j) % 3 != 0 else None for j in range(n_cols)]
    # Both kinds of missing value occur.
    assert any(not v for e in entries for v in e[entry_valid])
    assert any(v and y is None for e in entries for v, y in zip(e[entry_valid], e['y']))


def test_columns_file_holds_each_columns_index_and_fields():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(n_cols=4), path)

    schema = decode_schema(read_manifest(path)['cols']['schema'])
    expected = pa.schema([
        field('__col_idx', pa.int32(), 1, nullable=False),
        field('col_idx', pa.int32(), 2, nullable=False),
        field('s', pa.string(), 3, nullable=False),
    ])
    assert schema.equals(expected, check_metadata=True), schema
    file = read_parquet_file(f'{path}/cols.parquet')
    assert file.schema_arrow.equals(schema, check_metadata=True)
    assert file.read().to_pylist() == [{'__col_idx': j, 'col_idx': j, 's': str(j)} for j in range(4)]


def test_partitions_table_lists_each_partitions_entry_files():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)

    schema = read_parquet_file(f'{path}/partitions.parquet').schema_arrow
    file_ref = pa.struct([pa.field('path', pa.string(), False), pa.field('container', pa.string(), False)])
    assert schema.names == ['partition_index', 'start', 'end', 'row_count', 'row_file', 'entry_files']
    entry_files = schema.field('entry_files')
    assert not entry_files.nullable
    assert entry_files.type.value_type.equals(schema.field('row_file').type)
    assert entry_files.type.value_type.equals(file_ref)
    for partition, row_file, (entry_file,) in data_files(path):
        assert partition['entry_files'][0]['container'] == 'parquet'
        assert partition['entry_files'][0]['path'].startswith(f'entries/part-{partition["partition_index"]}-block-0-')
        assert entry_file.metadata.num_rows == row_file.metadata.num_rows == partition['row_count']


def row_group_sizes(file) -> list:
    return [file.metadata.row_group(i).num_rows for i in range(file.metadata.num_row_groups)]


def max_leaf_values(file) -> int:
    """The most values any leaf field of ``file`` holds in one row group."""
    metadata = file.metadata
    return max(
        metadata.row_group(i).column(j).num_values
        for i in range(metadata.num_row_groups)
        for j in range(metadata.num_columns)
    )


def test_row_groups_end_at_the_leaf_cap_in_row_and_entry_files_alike():
    # 5 rows per partition, and 5 values per row in each leaf of the entry file, so a leaf cap of 12
    # fits 2 rows in a row group.
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(n_rows=10, n_cols=5, n_partitions=2), path, leaf_cap=12)

    for _, row_file, (entry_file,) in data_files(path):
        assert row_group_sizes(row_file) == row_group_sizes(entry_file) == [2, 2, 1]
        assert max_leaf_values(entry_file) <= 12
        for i in range(entry_file.metadata.num_row_groups):
            entry_file.read_row_group(i)


def test_row_groups_end_at_the_leaf_cap_of_a_nested_entry_field():
    # Row i has 2 entries holding arrays of length i, so the array's leaf holds 2 * max(i, 1) values
    # for row i: 2, 2, 4, 6 and 8 for rows 0 to 4. A leaf cap of 8 groups them as [0, 1, 2], [3], [4].
    mt = hl.utils.range_matrix_table(5, 2, n_partitions=1)
    mt = mt.annotate_entries(a=hl.range(mt.row_idx))
    path = new_temp_file(extension='mt')
    write_columnar(mt, path, leaf_cap=8)

    ((_, row_file, (entry_file,)),) = data_files(path)
    assert row_group_sizes(row_file) == row_group_sizes(entry_file) == [3, 1, 1]
    assert max_leaf_values(entry_file) == 8
    entries = [e for i in range(entry_file.metadata.num_row_groups) for e in entry_file.read_row_group(i).to_pylist()]
    assert [e['a'] for e in entries] == [[list(range(i))] * 2 for i in range(5)]


def test_row_groups_end_at_the_byte_cap_of_a_string_entry_field():
    # Each row has 2 entries holding the same 4-byte string, which dictionary encoding stores in far
    # fewer bytes, so the target size never ends a row group. The string's leaf holds 8 bytes per
    # row, and a byte cap of 20 fits 2 rows in a row group.
    mt = hl.utils.range_matrix_table(5, 2, n_partitions=1)
    mt = mt.annotate_entries(s='PASS')
    path = new_temp_file(extension='mt')
    write_columnar(mt, path, byte_cap=20)

    ((_, row_file, (entry_file,)),) = data_files(path)
    assert row_group_sizes(row_file) == row_group_sizes(entry_file) == [2, 2, 1]
    assert [e['s'] for e in read_row_groups(entry_file).to_pylist()] == [['PASS'] * 2] * 5


def test_the_byte_cap_counts_each_string_leaf_of_a_struct_entry_field_on_its_own():
    # Each row has 3 entries of a struct of 2 strings of 2 bytes each: 6 bytes per row in each
    # string's leaf, so a byte cap of 12 fits 2 rows in a row group, though the struct holds 12
    # bytes per row in all.
    mt = hl.utils.range_matrix_table(3, 3, n_partitions=1)
    mt = mt.select_entries(t=hl.struct(u='ab', v='cd'))
    path = new_temp_file(extension='mt')
    write_columnar(mt, path, byte_cap=12)

    ((_, row_file, (entry_file,)),) = data_files(path)
    assert row_group_sizes(row_file) == row_group_sizes(entry_file) == [2, 1]
    assert [e['t'] for e in read_row_groups(entry_file).to_pylist()] == [[{'u': 'ab', 'v': 'cd'}] * 3] * 3


@pytest.mark.parametrize(
    'n_cols, entry, caps, message',
    [
        # 4 values per row in every leaf.
        (
            4,
            lambda mt: hl.struct(x=mt.col_idx),
            {'leaf_cap': 3},
            "more than 3 values in each entry field.*there are 4 columns",
        ),
        # Row 2 has 2 arrays of length 2, which is 4 values in the array's leaf.
        (2, lambda mt: hl.struct(a=hl.range(mt.row_idx)), {'leaf_cap': 3}, "more than 3 values in entry field 'a'"),
        # 3 entries of a struct whose first string has 4 bytes, which is 12 bytes in its leaf.
        (
            3,
            lambda mt: hl.struct(t=hl.struct(u='abcd', v='x')),
            {'byte_cap': 10},
            "more than 10 bytes of strings or binaries in entry field 't'",
        ),
    ],
)
def test_a_row_whose_entries_exceed_a_cap_fails_the_write(n_cols, entry, caps, message):
    mt = hl.utils.range_matrix_table(3, n_cols, n_partitions=1)
    mt = mt.select_entries(**entry(mt))
    path = new_temp_file(extension='mt')
    with pytest.raises(FatalError, match=message):
        write_columnar(mt, path, **caps)
    assert not is_committed(path)


@pytest.mark.parametrize(
    'annotate, message',
    [
        (lambda mt: mt.annotate_rows(**{'__row_uid': mt.row_idx}), "row field '__row_uid' can't be stored"),
        (lambda mt: mt.annotate_cols(**{'__col_uid': mt.col_idx}), "column field '__col_uid' can't be stored"),
    ],
)
def test_a_field_named_like_a_uid_fails_the_write(annotate, message):
    mt = annotate(hl.utils.range_matrix_table(2, 2, n_partitions=1))
    path = new_temp_file(extension='mt')
    with pytest.raises(FatalError, match=message):
        write_columnar(mt, path)
    assert not is_committed(path)


def hail_entries(mt: hl.MatrixTable) -> list:
    """The present entries of ``mt``, each with its row's and its column's fields, as Hail gives them."""
    return [to_python(e) for e in mt.entries().collect()]


def read_row_groups(file) -> pa.Table:
    """Reads ``file`` one whole row group at a time."""
    return pa.concat_tables([file.read_row_group(i) for i in range(file.metadata.num_row_groups)])


def joined_entries(path: str) -> list:
    """The present entries of the columnar matrix table at ``path``, each with its row's and its
    column's fields, read with pyarrow alone. Entries are joined to columns by column index and to
    rows by Row ID, as a generic engine would."""
    manifest = read_manifest(path)
    internal = manifest['internal_fields']
    row_id, col_idx, entry_valid = internal['row_id'], internal['col_idx'], internal['entry_valid']
    (block,) = manifest['entries']['column_blocks']
    width = block['end'] - block['start']

    files = data_files(path)
    rows = pa.concat_tables([read_row_groups(row_file) for _, row_file, _ in files])
    entries = pa.concat_tables([read_row_groups(entry_file) for _, _, (entry_file,) in files]).combine_chunks()
    entry_fields = [n for n in entries.column_names if n not in (row_id, entry_valid)]

    # One row per (row, column), at list position j of its row's lists.
    valid = entries[entry_valid].chunk(0)
    assert pc.all(pc.equal(pc.list_value_length(valid), width)).as_py()
    parents = pc.list_parent_indices(valid)
    positions = pc.subtract(pa.array(np.arange(len(parents))), pc.multiply(parents, width))
    exploded = pa.table({
        row_id: pc.take(entries[row_id], parents),
        col_idx: pc.cast(pc.add(positions, block['start']), pa.int32()),
        entry_valid: pc.list_flatten(valid),
        **{f: pc.list_flatten(entries[f].chunk(0)) for f in entry_fields},
    })
    present = exploded.filter(exploded[entry_valid]).drop_columns([entry_valid])

    cols = read_row_groups(read_parquet_file(f'{path}/cols.parquet'))
    joined = present.join(rows, row_id).join(cols, col_idx)
    return joined.sort_by([(row_id, 'ascending'), (col_idx, 'ascending')]).drop_columns([row_id, col_idx]).to_pylist()


def test_joining_entries_to_rows_and_columns_gives_the_matrix_tables_entries():
    mt = matrix_table(n_rows=12, n_cols=5, n_partitions=3)
    path = new_temp_file(extension='mt')
    # A small leaf cap, so that files have several row groups.
    write_columnar(mt, path, leaf_cap=10)

    assert joined_entries(path) == hail_entries(mt)


MISSING_ENTRIES_FIXTURE = 'columnar/0.1/matrix_table/missing_entries'


def test_output_matches_the_missing_entries_fixture():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)

    assert_same_dataset(path, resource(MISSING_ENTRIES_FIXTURE))


def test_missing_entries_fixture_holds_the_matrix_tables_entries():
    assert joined_entries(resource(MISSING_ENTRIES_FIXTURE)) == [
        to_python(e) for e in matrix_table().entries().collect()
    ]


def test_internal_field_names_do_not_collide_with_user_fields():
    mt = hl.utils.range_matrix_table(2, 2, n_partitions=1)
    mt = mt.annotate_rows(__row_id=1)
    # The Row ID field is in entry files too, so it must not collide with entry fields either.
    mt = mt.annotate_entries(__row_id1=2, __entry_valid=True)
    mt = mt.annotate_cols(__col_idx=3)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    assert read_manifest(path)['internal_fields'] == {
        'row_id': '__row_id2',
        'col_idx': '__col_idx1',
        'entry_valid': '__entry_valid1',
    }
    assert joined_entries(path) == hail_entries(mt)


def test_manifest_defines_the_reference_genomes_of_column_and_entry_fields():
    rg = custom_reference_genome()
    mt = hl.utils.range_matrix_table(2, 2, n_partitions=1)
    mt = mt.annotate_cols(l=hl.locus('1', mt.col_idx + 1, reference_genome='GRCh37'))
    mt = mt.annotate_entries(l2=hl.locus('chr1', mt.row_idx + 1, reference_genome=rg))
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    assert read_manifest(path)['reference_genomes'] == [CUSTOM_REFERENCE_GENOME]


def test_an_interpreted_write_records_the_row_key():
    mt = matrix_table(n_rows=9, n_cols=2, n_partitions=3)
    mt = mt.annotate_rows(a=mt.row_idx // 4).key_rows_by('a', 'row_idx')
    # A shuffle has no lowered implementation, so the write is interpreted rather than lowered.
    mt = mt.repartition(3).key_rows_by('a')
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    manifest = read_manifest(path)
    assert manifest['rows']['key'] == ['a']
    assert manifest['rows']['distinctly_keyed'] is False
    partitions = [p for p, _, _ in data_files(path) if p['row_count'] > 0]
    assert [(p['start'], p['end']) for p in partitions] == [({'a': a}, {'a': a}) for a in range(3)]

    # entries() orders entries by row key and then column key, so rows sharing a key interleave.
    def by_position(entries: list) -> list:
        return sorted(entries, key=lambda e: (e['row_idx'], e['col_idx']))

    assert by_position(joined_entries(path)) == by_position(hail_entries(mt))


def test_a_matrix_table_with_no_columns_has_one_empty_column_block():
    mt = hl.utils.range_matrix_table(3, 0, n_partitions=1)
    mt = mt.annotate_entries(x=mt.row_idx)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    manifest = read_manifest(path)
    assert manifest['cols']['count'] == 0
    assert manifest['entries']['column_blocks'] == [{'start': 0, 'end': 0}]
    assert read_parquet_file(f'{path}/cols.parquet').read().num_rows == 0
    ((_, _, (entry_file,)),) = data_files(path)
    entry_valid = manifest['internal_fields']['entry_valid']
    assert [(e[entry_valid], e['x']) for e in entry_file.read().to_pylist()] == [([], [])] * 3


@pytest.mark.parametrize('case', TYPE_CASES.keys())
def test_column_and_entry_fields_have_the_specified_extension_and_encoding(case):
    c = TYPE_CASES[case]()
    # Column j, and every entry in it, holds c.values[j].
    mt = hl.utils.range_matrix_table(2, len(c.values), n_partitions=1)
    mt = mt.annotate_cols(cx=required_by_index(mt.col_idx, c.values, c.typ))
    mt = mt.annotate_entries(x=mt.cx)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    # The manifest schemas, which equal the files' ARROW:schema. pyarrow's Parquet reader drops the
    # extension type of a required ndarray from a file's schema_arrow.
    manifest = read_manifest(path)
    assert_has_type(decode_schema(manifest['cols']['schema']).field('cx'), c)
    element = decode_schema(manifest['entries']['schema']).field('x').type.value_field
    assert element.nullable
    assert_has_type(element, c)

    cols = read_parquet_file(f'{path}/cols.parquet')
    assert [r['cx'] for r in cols.read().to_pylist()] == c.stored
    ((_, _, (entry_file,)),) = data_files(path)
    assert [e['x'] for e in entry_file.read().to_pylist()] == [c.stored] * 2


def read_columnar(path: str) -> hl.MatrixTable:
    return hl.MatrixTable(ir.MatrixRead(ir.MatrixColumnarReader(path)))


def assert_reads_as(path: str, expected: hl.MatrixTable) -> None:
    """The columnar matrix table at ``path`` reads as ``expected``: the same keys, types and values, with
    rows and columns in the same order."""
    actual = read_columnar(path)
    assert list(actual.row_key) == list(expected.row_key)
    assert list(actual.col_key) == list(expected.col_key)
    assert actual._same(expected)

    def keys(mt: hl.MatrixTable) -> tuple:
        return mt.aggregate_rows(hl.agg.collect(mt.row_key)), mt.aggregate_cols(hl.agg.collect(mt.col_key))

    assert keys(actual) == keys(expected)


def test_missing_entries_fixture_reads_as_the_matrix_table_it_holds():
    assert_reads_as(resource(MISSING_ENTRIES_FIXTURE), matrix_table())


def test_round_trip_preserves_values():
    mt = matrix_table(n_rows=12, n_cols=5, n_partitions=3)
    path = new_temp_file(extension='mt')
    # A small leaf cap, so that files have several row groups.
    write_columnar(mt, path, leaf_cap=10)

    assert_reads_as(path, mt)


@pytest.mark.parametrize('case', TYPE_CASES.keys())
def test_round_trip_of_column_and_entry_fields_preserves_values(case):
    c = TYPE_CASES[case]()
    # Column j, and every entry in it, holds c.values[j]. Entry (i, j) is missing when i + j is odd,
    # and its field x_opt when i + j is a multiple of 3.
    mt = hl.utils.range_matrix_table(3, len(c.values), n_partitions=2)
    mt = mt.annotate_cols(cx=required_by_index(mt.col_idx, c.values, c.typ))
    mt = mt.annotate_entries(
        x=mt.cx,
        x_opt=hl.or_missing((mt.row_idx + mt.col_idx) % 3 != 0, mt.cx),
        x_arr=hl.array([mt.cx, hl.missing(c.typ)]),
    )
    mt = mt.filter_entries((mt.row_idx + mt.col_idx) % 2 == 0)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    assert_reads_as(path, mt)


def localized_entries(mt: hl.MatrixTable) -> list:
    """Each row's entries, with ``None`` for a missing entry."""
    return [[None if e is None else dict(e) for e in r.es] for r in mt.localize_entries('es').collect()]


def test_missing_entries_stay_distinct_from_present_entries_with_missing_fields():
    n_rows, n_cols = 6, 4
    mt = read_columnar(resource(MISSING_ENTRIES_FIXTURE))

    # matrix_table's definition: entry (i, j) is missing when (i + j) % 4 == 1, and its field y is
    # missing when (i + j) % 3 == 0.
    assert localized_entries(mt) == [
        [
            None if (i + j) % 4 == 1 else {'x': i * 10 + j, 'y': None if (i + j) % 3 == 0 else str(j)}
            for j in range(n_cols)
        ]
        for i in range(n_rows)
    ]


def test_columns_come_from_the_columns_file_and_globals_from_the_globals_file():
    mt = read_columnar(resource(MISSING_ENTRIES_FIXTURE))

    assert mt.cols().collect() == [hl.Struct(col_idx=j, s=str(j)) for j in range(4)]
    assert hl.eval(mt.globals) == hl.Struct(g=5)
    assert mt.count_cols() == 4


def entry_file_paths(path: str) -> list:
    return [f'{path}/{f["path"]}' for p, _, _ in data_files(path) for f in p['entry_files']]


def test_entry_files_are_not_opened_when_no_entry_fields_are_requested():
    mt = matrix_table()
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)
    for file_path in entry_file_paths(path):
        hfs.remove(file_path)
        drop_checksum(file_path)

    actual = read_columnar(path)
    assert actual.rows().collect() == mt.rows().collect()
    assert actual.cols().collect() == mt.cols().collect()
    assert actual.count() == (6, 4)
    with pytest.raises(FatalError):
        actual.entries().collect()


def test_projecting_entry_fields_reads_only_those_fields_lists():
    mt = matrix_table()
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)
    internal = read_manifest(path)['internal_fields']
    for file_path in entry_file_paths(path):
        corrupt_column_chunks_except(
            file_path, {internal['row_id'], f'{internal["entry_valid"]}.list.element', 'x.list.element'}
        )

    def project(m: hl.MatrixTable) -> list:
        return localized_entries(m.select_entries('x'))

    assert project(read_columnar(path)) == project(mt)
    # Reading any other entry field reads a zeroed chunk.
    with pytest.raises(FatalError):
        read_columnar(path).select_entries('y').entries().collect()


def test_reading_only_whether_entries_are_missing_reads_only_the_entry_validity_list():
    mt = matrix_table()
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)
    internal = read_manifest(path)['internal_fields']
    for file_path in entry_file_paths(path):
        corrupt_column_chunks_except(file_path, {internal['row_id'], f'{internal["entry_valid"]}.list.element'})

    def project(m: hl.MatrixTable) -> list:
        return localized_entries(m.select_entries())

    assert project(read_columnar(path)) == project(mt)


def test_an_entry_field_named_like_the_row_uid_round_trips():
    mt = hl.utils.range_matrix_table(3, 2, n_partitions=1)
    mt = mt.annotate_entries(__row_uid=mt.row_idx * 10 + mt.col_idx)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    assert_reads_as(path, mt)


def rewrite_entry_file(path: str, f) -> None:
    """Rewrites the matrix table's first entry file with pyarrow, replacing its table ``t`` with ``f(t)``.
    The file keeps its Arrow schema."""
    file_path = entry_file_paths(path)[0]
    table = read_parquet_file(file_path).read()
    with hfs.open(file_path, 'wb') as out:
        pq.write_table(f(table), out, store_schema=True)
    drop_checksum(file_path)


def test_a_row_id_mismatch_fails_the_read():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)
    row_id = read_manifest(path)['internal_fields']['row_id']

    def shift_row_ids(t: pa.Table) -> pa.Table:
        i = t.schema.get_field_index(row_id)
        return t.set_column(i, t.schema.field(i), pc.add(t[row_id], 1))

    rewrite_entry_file(path, shift_row_ids)
    with pytest.raises(FatalError, match='Row ID'):
        read_columnar(path).entries().collect()


def test_an_entry_file_with_fewer_rows_than_its_row_file_fails_the_read():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)

    rewrite_entry_file(path, lambda t: t.slice(0, t.num_rows - 1))
    with pytest.raises(FatalError, match='length mismatch'):
        read_columnar(path).entries().collect()


def test_an_entry_list_of_the_wrong_length_fails_the_read():
    path = new_temp_file(extension='mt')
    write_columnar(matrix_table(), path)

    def drop_last_x(t: pa.Table) -> pa.Table:
        i = t.schema.get_field_index('x')
        x = pa.array([v[:-1] for v in t['x'].to_pylist()], t.schema.field(i).type)
        return t.set_column(i, t.schema.field(i), x)

    rewrite_entry_file(path, drop_last_x)
    with pytest.raises(FatalError, match='4 columns'):
        read_columnar(path).entries().collect()


def test_round_trip_of_a_matrix_table_with_no_columns():
    mt = hl.utils.range_matrix_table(3, 0, n_partitions=1)
    mt = mt.annotate_entries(x=mt.row_idx)
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)

    assert_reads_as(path, mt)


def test_reading_defines_the_matrix_tables_reference_genomes():
    name = f'ColumnarMatrixRead{secrets.token_hex(4)}'
    rg = hl.ReferenceGenome(name, ['b', 'a'], {'b': 100, 'a': 50})
    mt = hl.utils.range_matrix_table(2, 2, n_partitions=1)
    mt = mt.annotate_entries(l=hl.locus(hl.literal(['a', 'b'])[mt.col_idx], mt.row_idx + 1, reference_genome=rg))
    path = new_temp_file(extension='mt')
    write_columnar(mt, path)
    Env.backend().remove_reference(name)

    try:
        actual = read_columnar(path)
        assert hl.get_reference(name).contigs == ['b', 'a']
        assert [(e.l.contig, e.l.position) for e in actual.entries().collect()] == [
            ('a', 1),
            ('b', 1),
            ('a', 2),
            ('b', 2),
        ]
    finally:
        Env.backend().remove_reference(name)


def test_a_table_is_not_a_matrix_table():
    with pytest.raises(FatalError, match='is a table, not a matrix table'):
        read_columnar(resource('columnar/0.1/table/keyed')).count()


def test_a_matrix_table_is_not_a_table():
    with pytest.raises(FatalError, match='is a matrix table, not a table'):
        hl.Table(ir.TableRead(ir.TableColumnarReader(resource(MISSING_ENTRIES_FIXTURE)))).count()

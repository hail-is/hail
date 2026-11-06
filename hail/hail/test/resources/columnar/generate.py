"""Generates the columnar format conformance fixtures in this directory.

The fixtures are written with pyarrow alone, directly from the columnar format specification
(``dev-docs/hail-query/columnar-format-spec.md``), so they don't depend on Hail's writer. Hail's
tests write the same tables and compare the output with these fixtures by value
(``hail/python/test/hail/table/test_columnar_format.py``).

Run from any directory, with pyarrow installed, to regenerate ``<spec version>/table/<fixture>``:

    python hail/hail/test/resources/columnar/generate.py
"""

import base64
import json
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

SPEC_VERSION = '0.1'
ROW_ID = '__row_id'


def with_field_ids(schema: pa.Schema) -> pa.Schema:
    """Assigns field IDs depth-first (pre-order) from 1, as spec section 4.6 requires."""
    next_id = 0

    def assign(field: pa.Field) -> pa.Field:
        nonlocal next_id
        next_id += 1
        metadata = {b'PARQUET:field_id': str(next_id).encode()}
        typ = field.type
        if pa.types.is_list(typ):
            typ = pa.list_(assign(typ.value_field))
        elif pa.types.is_struct(typ):
            typ = pa.struct([assign(typ.field(i)) for i in range(typ.num_fields)])
        elif not (pa.types.is_primitive(typ) or pa.types.is_string(typ) or pa.types.is_binary(typ)):
            raise NotImplementedError(typ)
        return pa.field(field.name, typ, field.nullable, metadata)

    return pa.schema([assign(f) for f in schema])


def encode_schema(schema: pa.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode('ascii')


def write_parquet(path: str, schema: pa.Schema, rows: list) -> None:
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        path,
        compression='zstd',
        write_statistics=True,
        write_page_index=True,
        store_schema=True,
    )


def write_table(path: str, row_schema: pa.Schema, key: list, partitions: list, global_schema: pa.Schema, globals_row):
    """Writes a table whose row partitions are `partitions`, each a list of rows."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(f'{path}/rows')

    file_schema = with_field_ids(pa.schema([pa.field(ROW_ID, pa.int64(), False), *row_schema]))
    key_type = pa.struct([row_schema.field(k) for k in key])
    bounds = [pa.field('start', key_type), pa.field('end', key_type)] if key else []
    partitions_schema = with_field_ids(
        pa.schema([
            pa.field('partition_index', pa.int32(), False),
            *bounds,
            pa.field('row_count', pa.int64(), False),
            pa.field(
                'row_file',
                pa.struct([pa.field('path', pa.string(), False), pa.field('container', pa.string(), False)]),
                False,
            ),
        ])
    )

    digits = len(str(len(partitions) - 1))
    entries = []
    distinctly_keyed = bool(key)
    for i, rows in enumerate(partitions):
        row_file = f'rows/part-{i:0{digits}}-00000000-0000-4000-8000-{i:012}.parquet'
        write_parquet(f'{path}/{row_file}', file_schema, [{ROW_ID: (i << 40) | j, **r} for j, r in enumerate(rows)])
        entry = {'partition_index': i}
        if key:
            keys = [tuple(r[k] for k in key) for r in rows]
            distinctly_keyed = distinctly_keyed and len(set(keys)) == len(keys)
            entry['start'] = {k: rows[0][k] for k in key} if rows else None
            entry['end'] = {k: rows[-1][k] for k in key} if rows else None
        entry['row_count'] = len(rows)
        entry['row_file'] = {'path': row_file, 'container': 'parquet'}
        entries.append(entry)
    write_parquet(f'{path}/partitions.parquet', partitions_schema, entries)

    global_schema = with_field_ids(global_schema)
    globals_file = None
    if len(global_schema) > 0:
        globals_file = {'path': 'globals.parquet', 'container': 'parquet'}
        write_parquet(f'{path}/globals.parquet', global_schema, [globals_row])

    manifest = {
        'format': 'hail-columnar',
        'version': SPEC_VERSION,
        'kind': 'table',
        'rows': {'schema': encode_schema(file_schema), 'key': key, 'distinctly_keyed': distinctly_keyed},
        'globals': {'schema': encode_schema(global_schema), 'file': globals_file},
        'partitions': {'path': 'partitions.parquet', 'container': 'parquet'},
        'internal_fields': {'row_id': ROW_ID},
        'reference_genomes': [],
    }
    # The manifest is written last.
    with open(f'{path}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


# Rows 0 to 5 in two row partitions of three rows, with every type the writer supports so far.
# Hail's tests build the same table from hl.utils.range_table(6, n_partitions=2).
BASIC_ROW_SCHEMA = pa.schema([
    pa.field('idx', pa.int32(), False),
    pa.field('s', pa.string(), False),
    pa.field('f64', pa.float64(), False),
    pa.field('f32', pa.float32()),
    pa.field('i64', pa.int64(), False),
    pa.field('b', pa.bool_(), False),
    pa.field('ms', pa.string()),
    pa.field('arr', pa.list_(pa.field('element', pa.int32())), False),
    pa.field('st', pa.struct([pa.field('x', pa.int32(), False), pa.field('y', pa.string())]), False),
])


def basic_row(i: int) -> dict:
    return {
        'idx': i,
        's': str(i),
        'f64': i / 2,
        'f32': None if i % 3 == 0 else float(i),
        'i64': i * 1_000_000_000_000,
        'b': i % 2 == 0,
        'ms': None,
        'arr': [None if j == 1 else j for j in range(i % 3)],
        'st': {'x': i, 'y': 'y' if i % 2 == 0 else None},
    }


BASIC_PARTITIONS = [[basic_row(i) for i in range(3)], [basic_row(i) for i in range(3, 6)]]

GLOBAL_SCHEMA = pa.schema([pa.field('g', pa.int32(), False), pa.field('gs', pa.string(), False)])
GLOBALS = {'g': 5, 'gs': 'hello'}


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), SPEC_VERSION, 'table')
    write_table(f'{root}/keyed', BASIC_ROW_SCHEMA, ['idx'], BASIC_PARTITIONS, GLOBAL_SCHEMA, GLOBALS)
    write_table(f'{root}/unkeyed', BASIC_ROW_SCHEMA, [], BASIC_PARTITIONS, GLOBAL_SCHEMA, GLOBALS)
    write_table(
        f'{root}/no_globals',
        pa.schema([pa.field('idx', pa.int32(), False), pa.field('s', pa.string(), False)]),
        ['idx'],
        [[{'idx': i, 's': str(i)} for i in range(3)], [{'idx': i, 's': str(i)} for i in range(3, 6)]],
        pa.schema([]),
        None,
    )


if __name__ == '__main__':
    main()

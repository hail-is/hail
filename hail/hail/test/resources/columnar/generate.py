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
    """Assigns field IDs depth-first (pre-order) from 1 to every field except a map's entries field, as
    spec section 4.6 requires. Existing field metadata, such as extension metadata, is kept."""
    next_id = 0

    def assign(field: pa.Field) -> pa.Field:
        nonlocal next_id
        next_id += 1
        metadata = {**(field.metadata or {}), b'PARQUET:field_id': str(next_id).encode()}
        typ = field.type
        if pa.types.is_fixed_size_list(typ):
            typ = pa.list_(assign(typ.value_field), typ.list_size)
        elif pa.types.is_list(typ):
            typ = pa.list_(assign(typ.value_field))
        elif pa.types.is_map(typ):
            # pyarrow gives the entries field no metadata, so no field ID.
            typ = pa.map_(assign(typ.key_field), assign(typ.item_field), keys_sorted=typ.keys_sorted)
        elif pa.types.is_struct(typ):
            typ = pa.struct([assign(typ.field(i)) for i in range(typ.num_fields)])
        elif not (pa.types.is_primitive(typ) or pa.types.is_string(typ) or pa.types.is_binary(typ)):
            raise NotImplementedError(typ)
        return pa.field(field.name, typ, field.nullable, metadata)

    return pa.schema([assign(f) for f in schema])


def extension_metadata(name: str, metadata: dict) -> dict:
    """Arrow field metadata marking a field as extension type `name`, with extension metadata `metadata`. pyarrow keeps these keys on the
    storage type's field, so a reader recovers the extension type from the footer's ARROW:schema."""
    return {
        b'ARROW:extension:name': name.encode(),
        b'ARROW:extension:metadata': json.dumps(metadata, separators=(',', ':')).encode(),
    }


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


def write_table(
    path: str,
    row_schema: pa.Schema,
    key: list,
    partitions: list,
    global_schema: pa.Schema,
    globals_row,
    reference_genomes: tuple = (),
):
    """Writes a table whose row partitions are `partitions`, each a list of rows. `reference_genomes`
    holds the definitions of its non-builtin reference genomes (spec section 3.1)."""
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
        'reference_genomes': list(reference_genomes),
    }
    # The manifest is written last.
    with open(f'{path}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')


# Rows 0 to 5 in two row partitions of three rows, with primitive, struct and array fields.
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


# A non-builtin reference genome (spec section 3.1) whose contig order is not lexicographic.
CUSTOM_REFERENCE_GENOME = {
    'name': 'ColumnarTestGenome',
    'contigs': [
        {'name': 'chr2', 'length': 1000},
        {'name': 'chr10', 'length': 1000},
        {'name': 'chr1', 'length': 1000},
        {'name': 'chrX', 'length': 500},
    ],
    'xContigs': ['chrX'],
    'yContigs': [],
    'mtContigs': [],
    'par': [{'start': {'contig': 'chrX', 'position': 1}, 'end': {'contig': 'chrX', 'position': 100}}],
}


def contig_names(reference: dict) -> list:
    return [c['name'] for c in reference['contigs']]


def grch37_contigs() -> list:
    """GRCh37's contigs in genome order, from the reference genome definition Hail ships."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, '..', '..', '..', 'resources', 'reference', 'grch37.json')) as f:
        return contig_names(json.load(f))


def locus(contigs: list, contig: str, position: int) -> int:
    """Spec section 5.2."""
    return (contigs.index(contig) << 32) | position


def call(alleles: list, phased: bool = False) -> int:
    """Spec section 5.3."""
    ploidy = len(alleles)
    if ploidy == 0:
        representation = 0
    elif ploidy == 1:
        representation = alleles[0]
    else:
        j, k = (alleles[0], alleles[0] + alleles[1]) if phased else sorted(alleles)
        representation = k * (k + 1) // 2 + j
    return (representation << 3) | (ploidy << 1) | int(phased)


def all_types() -> list:
    """For each type the columnar format can store, except binary, which Hail's Python API can't make: its
    name, its Arrow storage type, its extension metadata, and its stored value in row `i`, for `i` in 0
    to 3. Hail's tests build the same values (``all_types_values`` in ``test_columnar_format.py``).

    Hail infers values inside a literal, such as struct fields and array elements, to be nullable, so
    the storage types say the same."""
    contigs = contig_names(CUSTOM_REFERENCE_GENOME)
    custom_locus = extension_metadata(
        'hail.locus', {'reference_genome': CUSTOM_REFERENCE_GENOME['name'], 'contigs': contigs}
    )
    grch37 = grch37_contigs()
    calls = [call([0, 1]), call([1, 1]), call([0, 1], phased=True), call([1])]
    return [
        ('bool', pa.bool_(), {}, lambda i: i % 2 == 0),
        ('int32', pa.int32(), {}, lambda i: i - 1),
        ('int64', pa.int64(), {}, lambda i: i * 1_000_000_000_000 - 1),
        ('float32', pa.float32(), {}, lambda i: i / 4),
        ('float64', pa.float64(), {}, lambda i: i * 1.5),
        ('str', pa.string(), {}, lambda i: f'v{i}'),
        (
            'struct',
            pa.struct([pa.field('a', pa.int32()), pa.field('b', pa.string())]),
            {},
            lambda i: {'a': i, 'b': str(i)},
        ),
        ('empty_struct', pa.bool_(), extension_metadata('hail.empty_struct', {'tuple': False}), lambda i: True),
        (
            'tuple',
            pa.struct([pa.field('0', pa.int32()), pa.field('1', pa.string())]),
            extension_metadata('hail.tuple', {}),
            lambda i: {'0': i, '1': str(i)},
        ),
        ('empty_tuple', pa.bool_(), extension_metadata('hail.empty_struct', {'tuple': True}), lambda i: True),
        ('array', pa.list_(pa.field('element', pa.int32())), {}, lambda i: list(range(i))),
        (
            'set',
            pa.list_(pa.field('element', pa.int32())),
            extension_metadata('hail.set', {}),
            lambda i: sorted({3 - j for j in range(i)}),
        ),
        (
            'dict',
            pa.map_(pa.field('key', pa.string(), False), pa.field('value', pa.int32()), keys_sorted=True),
            {},
            lambda i: sorted((str(3 - j), None if j == 1 else j) for j in range(i)),
        ),
        ('locus', pa.int64(), custom_locus, lambda i: locus(contigs, contigs[i], i + 1)),
        (
            'builtin_locus',
            pa.int64(),
            extension_metadata('hail.locus', {'reference_genome': 'GRCh37', 'contigs': grch37}),
            lambda i: locus(grch37, str(i + 1), 100 + i),
        ),
        (
            'interval',
            pa.struct([
                pa.field('start', pa.int64(), metadata=custom_locus),
                pa.field('end', pa.int64(), metadata=custom_locus),
                pa.field('includes_start', pa.bool_(), False),
                pa.field('includes_end', pa.bool_(), False),
            ]),
            extension_metadata('hail.interval', {}),
            lambda i: {
                'start': locus(contigs, contigs[i], 1),
                'end': locus(contigs, contigs[i], 10 + i),
                'includes_start': True,
                'includes_end': i % 2 == 0,
            },
        ),
        ('call', pa.int32(), extension_metadata('hail.call', {'encoding': 1}), lambda i: calls[i]),
        (
            'ndarray',
            # shape is nullable, though never null (spec section 5.4).
            pa.struct([
                pa.field('data', pa.list_(pa.field('element', pa.float64(), False)), False),
                pa.field('shape', pa.list_(pa.field('element', pa.int32(), False), 2)),
            ]),
            extension_metadata('arrow.variable_shape_tensor', {}),
            # Row-major data of an i x 2 ndarray holding 0, 1, 2, ...
            lambda i: {'data': [float(k) for k in range(2 * i)], 'shape': [i, 2]},
        ),
    ]


def all_types_table() -> tuple:
    """The row schema and row partitions of the `all_types` fixture: rows 0 to 3 in two row partitions.
    For each type, row `i` has a required field, a nullable field that is missing in odd rows, and a
    struct holding both, and arrays of them with required and with nullable elements."""
    fields = [pa.field('idx', pa.int32(), False)]
    for name, storage, metadata, _ in all_types():

        def f(field_name, nullable):
            return pa.field(field_name, storage, nullable, metadata)

        fields += [
            f(name, False),
            f(f'{name}_opt', True),
            pa.field(
                f'{name}_nested',
                pa.struct([
                    f('req', False),
                    f('opt', True),
                    pa.field('req_arr', pa.list_(f('element', False)), False),
                    pa.field('opt_arr', pa.list_(f('element', True)), False),
                ]),
                False,
            ),
        ]

    def row(i):
        r = {'idx': i}
        for name, _, _, value in all_types():
            x = value(i)
            x_opt = x if i % 2 == 0 else None
            r[name] = x
            r[f'{name}_opt'] = x_opt
            r[f'{name}_nested'] = {'req': x, 'opt': x_opt, 'req_arr': [x], 'opt_arr': [x, None]}
        return r

    return pa.schema(fields), [[row(0), row(1)], [row(2), row(3)]]


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), SPEC_VERSION, 'table')
    all_types_schema, all_types_partitions = all_types_table()
    write_table(
        f'{root}/all_types',
        all_types_schema,
        ['idx'],
        all_types_partitions,
        GLOBAL_SCHEMA,
        GLOBALS,
        reference_genomes=(CUSTOM_REFERENCE_GENOME,),
    )
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

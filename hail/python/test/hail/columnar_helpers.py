"""Helpers for the conformance tests of the columnar format, which read Hail's output with pyarrow,
independently of Hail (``table/test_columnar_format.py`` and ``matrixtable/test_columnar_format.py``)."""

import base64
import io
import json
import struct
from typing import NamedTuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import hail as hl
import hailtop.fs as hfs

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


def read_manifest(path: str) -> dict:
    with hfs.open(f'{path}/manifest.json') as f:
        return json.load(f)


def read_parquet_file(path: str) -> pq.ParquetFile:
    with hfs.open(path, 'rb') as f:
        return pq.ParquetFile(io.BytesIO(f.read()))


def to_python(value):
    """A collected Hail value as the plain Python value pyarrow's ``to_pylist`` gives."""
    if isinstance(value, hl.Struct):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    return value


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


EXTENSION_NAME = b'ARROW:extension:name'
EXTENSION_METADATA = b'ARROW:extension:metadata'


def parse_extension(f: pa.Field) -> tuple:
    """The extension name and parsed metadata of a field whose extension type pyarrow doesn't know, which
    pyarrow leaves in the Arrow field metadata. ``(None, None)`` for a field with no extension type."""
    metadata = f.metadata or {}
    if EXTENSION_NAME not in metadata:
        return None, None
    return metadata[EXTENSION_NAME].decode(), json.loads(metadata[EXTENSION_METADATA])


def required_by_index(i: hl.Int32Expression, values: list, typ: hl.HailType) -> hl.Expression:
    """``values[i]``, as an expression Hail infers to be required. Indexing a literal array would be
    inferred optional."""
    if isinstance(typ, (hl.tstruct, hl.ttuple)) and len(typ) == 0:
        # hl.switch rejects branches of empty struct or tuple type as having different types. Every value
        # of such a type is the same.
        return hl.literal(values[0], typ)
    switch = hl.switch(i)
    for j, v in enumerate(values[:-1]):
        switch = switch.when(j, hl.literal(v, typ))
    return switch.default(hl.literal(values[-1], typ))


class TypeCase(NamedTuple):
    """A Hail type, values of it, and the Arrow storage type, extension name, extension metadata and stored
    values that spec section 5 gives for them.

    pyarrow parses the metadata of a canonical extension type itself and doesn't expose it. For such a
    type, ``extension_metadata`` is None and ``canonical_type`` is the type pyarrow parses the field to,
    whose repr shows the parameters it parsed."""

    typ: hl.HailType
    values: list
    storage: pa.DataType
    extension_name: Optional[str]
    extension_metadata: Optional[dict]
    stored: list
    canonical_type: Optional[str] = None


# Each case is built when its test runs, since building some cases initializes Hail.
TYPE_CASES = {
    'call': lambda: TypeCase(
        hl.tcall,
        [hl.Call([0, 1]), hl.Call([1, 1]), hl.Call([0, 1], phased=True), hl.Call([1]), hl.Call([])],
        pa.int32(),
        'hail.call',
        {'encoding': 1},
        # (allele representation << 3) | (ploidy << 1) | phased, from spec section 5.3.
        [(1 << 3) | 4, (2 << 3) | 4, (1 << 3) | 4 | 1, (1 << 3) | 2, 0],
    ),
    'locus': lambda: TypeCase(
        hl.tlocus('GRCh37'),
        [hl.Locus('1', 5), hl.Locus('2', 10), hl.Locus('X', 7), hl.Locus('MT', 16569)],
        pa.int64(),
        'hail.locus',
        {'reference_genome': 'GRCh37', 'contigs': hl.get_reference('GRCh37').contigs},
        # (contig_index << 32) | position, from spec section 5.2. X and MT are GRCh37's 23rd and 25th contigs.
        [(0 << 32) | 5, (1 << 32) | 10, (22 << 32) | 7, (24 << 32) | 16569],
    ),
    'set': lambda: TypeCase(
        hl.tset(hl.tint32),
        [{3, 1, 2}, set(), {5}],
        # Hail infers a literal's set elements to be optional.
        pa.list_(pa.field('element', pa.int32())),
        'hail.set',
        {},
        [[1, 2, 3], [], [5]],
    ),
    'tuple': lambda: TypeCase(
        hl.ttuple(hl.tint32, hl.tstr),
        [(1, 'a'), (2, None), (3, 'c')],
        pa.struct([pa.field('0', pa.int32()), pa.field('1', pa.string())]),
        'hail.tuple',
        {},
        [{'0': 1, '1': 'a'}, {'0': 2, '1': None}, {'0': 3, '1': 'c'}],
    ),
    # A present empty struct or tuple is true, and a missing one null (spec section 5.5).
    'empty_struct': lambda: TypeCase(
        hl.tstruct(),
        [hl.Struct(), hl.Struct(), hl.Struct()],
        pa.bool_(),
        'hail.empty_struct',
        {'tuple': False},
        [True, True, True],
    ),
    'empty_tuple': lambda: TypeCase(
        hl.ttuple(),
        [(), (), ()],
        pa.bool_(),
        'hail.empty_struct',
        {'tuple': True},
        [True, True, True],
    ),
    'interval': lambda: TypeCase(
        hl.tinterval(hl.tint32),
        [hl.Interval(1, 5, True, False), hl.Interval(2, 3, False, True), hl.Interval(7, 7, True, True)],
        pa.struct([
            pa.field('start', pa.int32()),
            pa.field('end', pa.int32()),
            pa.field('includes_start', pa.bool_(), False),
            pa.field('includes_end', pa.bool_(), False),
        ]),
        'hail.interval',
        {},
        [
            {'start': 1, 'end': 5, 'includes_start': True, 'includes_end': False},
            {'start': 2, 'end': 3, 'includes_start': False, 'includes_end': True},
            {'start': 7, 'end': 7, 'includes_start': True, 'includes_end': True},
        ],
    ),
    'dict': lambda: TypeCase(
        hl.tdict(hl.tstr, hl.tint32),
        [{'b': 2, 'a': None}, {}, {'c': 3}],
        pa.map_(pa.field('key', pa.string(), False), pa.field('value', pa.int32()), keys_sorted=True),
        None,
        None,
        [[('a', None), ('b', 2)], [], [('c', 3)]],
    ),
    'ndarray': lambda: TypeCase(
        hl.tndarray(hl.tfloat64, 2),
        [np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.zeros((0, 2)), np.array([[7.0]])],
        # pyarrow replaces the storage type of this canonical extension type with its own, which names
        # list elements 'item' and makes every field nullable. ArrowSchemaConversionSuite and
        # ColumnarSchemaSuite check the stored layout; that x_opt reads at all shows that shape is
        # nullable (spec section 5.4).
        pa.struct([pa.field('data', pa.list_(pa.float64())), pa.field('shape', pa.list_(pa.int32(), 2))]),
        'arrow.variable_shape_tensor',
        None,
        [
            {'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'shape': [2, 3]},
            {'data': [], 'shape': [0, 2]},
            {'data': [7.0], 'shape': [1, 1]},
        ],
        # No dim_names, permutation or uniform_shape.
        canonical_type='extension<arrow.variable_shape_tensor[value_type=double, ndim=2]>',
    ),
}


# A non-builtin reference genome, defined as spec section 3.1 says. Its contig order is not
# lexicographic: chr2 < chr10 < chr1 < chrX.
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


def custom_reference_genome() -> hl.ReferenceGenome:
    """``CUSTOM_REFERENCE_GENOME``, defined in Hail."""
    definition = CUSTOM_REFERENCE_GENOME
    return hl.ReferenceGenome(
        definition['name'],
        [c['name'] for c in definition['contigs']],
        {c['name']: c['length'] for c in definition['contigs']},
        x_contigs=definition['xContigs'],
        y_contigs=definition['yContigs'],
        mt_contigs=definition['mtContigs'],
        par=[(p['start']['contig'], p['start']['position'], p['end']['position']) for p in definition['par']],
    )


def without_schemas(manifest: dict) -> dict:
    return {k: {**v, 'schema': None} if isinstance(v, dict) and 'schema' in v else v for k, v in manifest.items()}


def data_files(path: str) -> list:
    """Each row partition's row in the partitions table, with its row file and, for a matrix table, its
    entry files."""
    partitions = read_parquet_file(f'{path}/partitions.parquet').read().to_pylist()
    return [
        (
            p,
            read_parquet_file(f'{path}/{p["row_file"]["path"]}'),
            [read_parquet_file(f'{path}/{f["path"]}') for f in p.get('entry_files', [])],
        )
        for p in partitions
    ]


def row_files(path: str) -> list:
    """Each row partition's row in the partitions table, with its row file."""
    return [(p, row_file) for p, row_file, _ in data_files(path)]


def assert_has_type(f: pa.Field, c: TypeCase) -> None:
    """``f`` has the Arrow storage type and extension type that spec section 5 gives for ``c``."""
    if c.canonical_type is None:
        assert parse_extension(f) == (c.extension_name, c.extension_metadata), f
        storage_type = f.type
    else:
        assert isinstance(f.type, pa.BaseExtensionType), f
        assert (f.type.extension_name, str(f.type)) == (c.extension_name, c.canonical_type), f
        storage_type = f.type.storage_type
    assert storage_type.equals(c.storage), f


def assert_same_file(actual: pq.ParquetFile, expected: pq.ParquetFile) -> None:
    """``actual`` has the same Arrow schema and the same rows as ``expected``."""
    assert actual.schema_arrow.equals(expected.schema_arrow, check_metadata=True)
    assert actual.read().to_pylist() == expected.read().to_pylist()


def assert_same_dataset(actual: str, expected: str) -> None:
    """Compares two columnar tables, or two columnar matrix tables, by value. Data file paths may
    differ."""
    actual_manifest = read_manifest(actual)
    expected_manifest = read_manifest(expected)
    assert without_schemas(actual_manifest) == without_schemas(expected_manifest)
    for component in ('rows', 'globals', 'cols', 'entries'):
        if component in expected_manifest:
            actual_schema = decode_schema(actual_manifest[component]['schema'])
            assert actual_schema.equals(decode_schema(expected_manifest[component]['schema']), check_metadata=True)

    actual_partitions = data_files(actual)
    expected_partitions = data_files(expected)
    assert len(actual_partitions) == len(expected_partitions)
    for (actual_partition, actual_rows, actual_entries), (expected_partition, expected_rows, expected_entries) in zip(
        actual_partitions, expected_partitions
    ):
        for p in (actual_partition, expected_partition):
            del p['row_file']['path']
            for f in p.get('entry_files', []):
                del f['path']
        assert actual_partition == expected_partition
        assert_same_file(actual_rows, expected_rows)
        assert len(actual_entries) == len(expected_entries)
        for a, e in zip(actual_entries, expected_entries):
            assert_same_file(a, e)

    partitions_schema = read_parquet_file(f'{actual}/partitions.parquet').schema_arrow
    assert partitions_schema.equals(
        read_parquet_file(f'{expected}/partitions.parquet').schema_arrow, check_metadata=True
    )
    files = ['globals.parquet'] if expected_manifest['globals']['file'] is not None else []
    if 'cols' in expected_manifest:
        files.append('cols.parquet')
    for file in files:
        assert_same_file(read_parquet_file(f'{actual}/{file}'), read_parquet_file(f'{expected}/{file}'))

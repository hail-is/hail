from typing import List, Optional

from hail.ir import (
    BlockMatrixBinaryMultiWriter,
    BlockMatrixMultiWrite,
    BlockMatrixNativeMultiWriter,
    BlockMatrixTextMultiWriter,
    MatrixMultiWrite,
    MatrixNativeMultiWriter,
    MatrixNativePartitionedColumnsWriter,
    MatrixWrite,
)
from hail.linalg import BlockMatrix
from hail.matrixtable import MatrixTable
from hail.typecheck import enumeration, nullable, sequenceof, typecheck
from hail.utils.java import Env


@typecheck(mts=sequenceof(MatrixTable), prefix=str, overwrite=bool, stage_locally=bool, codec_spec=nullable(str))
def write_matrix_tables(
    mts: List[MatrixTable], prefix: str, overwrite: bool = False, stage_locally: bool = False, codec_spec=None
):
    length = len(str(len(mts) - 1))
    paths = [f"{prefix}{str(i).rjust(length, '0')}.mt" for i in range(len(mts))]
    writer = MatrixNativeMultiWriter(paths, overwrite, codec_spec)
    Env.backend().execute(MatrixMultiWrite([mt._mir for mt in mts], writer))

    return paths


@typecheck(bms=sequenceof(BlockMatrix), prefix=str, overwrite=bool)
def block_matrices_tofiles(bms: List[BlockMatrix], prefix: str, overwrite: bool = False):
    writer = BlockMatrixBinaryMultiWriter(prefix, overwrite)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))


@typecheck(mt=MatrixTable, prefix=str, n_fanout_targets=int, overwrite=bool, codec_spec=nullable(str))
def write_mts_split_by_cols(
    mt: MatrixTable, prefix: str, n_fanout_targets: int = 50, overwrite: bool = False, codec_spec: str | None = None
):
    fanout_limit = 100  # magic number from MatrixNativePartitionedColumnsWriter.scala, keep in sync
    if not 1 < n_fanout_targets <= fanout_limit:
        raise ValueError(f'fanout limit must be between 2 and {fanout_limit}, got {n_fanout_targets}')

    from hail import current_backend

    current_backend().validate_file(prefix)
    writer = MatrixNativePartitionedColumnsWriter(prefix, n_fanout_targets, overwrite, codec_spec)
    Env.backend().execute(MatrixWrite(mt._mir, writer))


@typecheck(
    bms=sequenceof(BlockMatrix),
    prefix=str,
    overwrite=bool,
    delimiter=str,
    header=nullable(str),
    add_index=bool,
    compression=nullable(enumeration('gz', 'bgz')),
    custom_filenames=nullable(sequenceof(str)),
)
def export_block_matrices(
    bms: List[BlockMatrix],
    prefix: str,
    overwrite: bool = False,
    delimiter: str = '\t',
    header: Optional[str] = None,
    add_index: bool = False,
    compression: Optional[str] = None,
    custom_filenames=None,
):
    if custom_filenames:
        assert len(custom_filenames) == len(bms), (
            "Number of block matrices and number of custom filenames must be equal"
        )

    writer = BlockMatrixTextMultiWriter(prefix, overwrite, delimiter, header, add_index, compression, custom_filenames)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))


@typecheck(bms=sequenceof(BlockMatrix), path_prefix=str, overwrite=bool, force_row_major=bool, stage_locally=bool)
def write_block_matrices(
    bms: List[BlockMatrix],
    path_prefix: str,
    overwrite: bool = False,
    force_row_major: bool = False,
    stage_locally: bool = False,
):
    """Writes a sequence of block matrices to disk in the same format as BlockMatrix.write.

    :param bms: :obj:`list` of :class:`BlockMatrix`
        Block matrices to write to disk.
    :param path_prefix: obj:`str`
        Prefix of path to write the block matrices to.
    :param overwrite: obj:`bool`
        If true, overwrite any files with the same name as the block matrices being generated.
    :param force_row_major: obj:`bool`
        If ``True``, transform blocks in column-major format
        to row-major format before writing.
        If ``False``, write blocks in their current format.
    :param stage_locally: :obj:`bool`
        Deprecated and ignored. Retained for backwards compatibility; has no
        effect.
    """
    writer = BlockMatrixNativeMultiWriter(path_prefix, overwrite, force_row_major)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))

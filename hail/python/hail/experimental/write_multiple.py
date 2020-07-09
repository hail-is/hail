from typing import List, Optional

from hail import MatrixTable
from hail.linalg import BlockMatrix
from hail.ir import MatrixMultiWrite, MatrixNativeMultiWriter, BlockMatrixMultiWrite, BlockMatrixBinaryMultiWriter, BlockMatrixTextMultiWriter, BlockMatrixNativeMultiWriter
from hail.typecheck import nullable, sequenceof, typecheck, enumeration
from hail.utils.java import Env


@typecheck(mts=sequenceof(MatrixTable),
           prefix=str,
           overwrite=bool,
           stage_locally=bool)
def write_matrix_tables(mts: List[MatrixTable], prefix: str, overwrite: bool = False,
                        stage_locally: bool = False):
    writer = MatrixNativeMultiWriter(prefix, overwrite, stage_locally)
    Env.backend().execute(MatrixMultiWrite([mt._mir for mt in mts], writer))


@typecheck(bms=sequenceof(BlockMatrix),
           prefix=str,
           overwrite=bool)
def block_matrices_tofiles(bms: List[BlockMatrix], prefix: str, overwrite: bool = False):
    writer = BlockMatrixBinaryMultiWriter(prefix, overwrite)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))


@typecheck(bms=sequenceof(BlockMatrix),
           prefix=str,
           overwrite=bool,
           delimiter=str,
           header=nullable(str),
           add_index=bool,
           compression=nullable(enumeration('gz', 'bgz')),
           custom_filenames=nullable(sequenceof(str)))
def export_block_matrices(bms: List[BlockMatrix], prefix: str, overwrite: bool = False,
                          delimiter: str = '\t', header: Optional[str] = None, add_index: bool = False,
                          compression: Optional[str] = None, custom_filenames=None):

    if custom_filenames:
        assert len(custom_filenames) == len(bms), "Number of block matrices and number of custom filenames must be equal"

    writer = BlockMatrixTextMultiWriter(prefix, overwrite, delimiter, header, add_index, compression, custom_filenames)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))


@typecheck(bms=sequenceof(BlockMatrix), path_prefix=str, overwrite=bool, force_row_major=bool, stage_locally=bool)
def write_block_matrices(bms: List[BlockMatrix], path_prefix: str, overwrite: bool = False,
                         force_row_major: bool = False, stage_locally: bool = False):
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
        If ``True``, major output will be written to temporary local storage
        before being copied to ``output``.
    """
    writer = BlockMatrixNativeMultiWriter(path_prefix, overwrite, force_row_major, stage_locally)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))

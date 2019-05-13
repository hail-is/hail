from typing import List, Optional

from hail import MatrixTable
from hail.linalg import BlockMatrix
from hail.ir import MatrixMultiWrite, MatrixNativeMultiWriter, BlockMatrixMultiWrite, BlockMatrixBinaryMultiWriter, BlockMatrixTextMultiWriter
from hail.typecheck import nullable, sequenceof, typecheck
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
           add_index=bool)
def export_block_matrices(bms: List[BlockMatrix], prefix: str, overwrite: bool = False,
                          delimiter: str = '\t', header: Optional[str] = None,  add_index: bool = False):
    writer = BlockMatrixTextMultiWriter(prefix, overwrite, delimiter, header, add_index)
    Env.backend().execute(BlockMatrixMultiWrite([bm._bmir for bm in bms], writer))

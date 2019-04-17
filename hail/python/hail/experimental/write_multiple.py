from typing import List

from hail import MatrixTable
from hail.linalg import BlockMatrix
from hail.ir import MatrixMultiWrite, MatrixNativeMultiWriter
from hail.typecheck import sequenceof, typecheck
from hail.utils.java import Env

@typecheck(mts=sequenceof(MatrixTable),
           prefix=str,
           overwrite=bool,
           stage_locally=bool)
def write_matrix_tables(mts: List[MatrixTable], prefix: str, overwrite: bool = False,
                        stage_locally: bool = False):
    writer = MatrixNativeMultiWriter(prefix, overwrite, stage_locally)
    Env.backend().execute(MatrixMultiWrite([mt._mir for mt in mts], writer))

@typecheck(mts=sequenceof(BlockMatrix),
           prefix=str,
           overwrite=bool,
           force_row_major=bool,
           stage_locally=bool)
def write_block_matrices(bms: List[BlockMatrix], prefix: str, overwrite: bool = False,
                         force_row_major: bool = False, stage_locally: bool = False):
    writer = BlockMatrixNativeWriter(prefix, overwrite, force_row_major, stage_locally)
    Env.backend().execute(MatrixMultiWrite([bm._bmir for bm in bms], writer))

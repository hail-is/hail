from typing import List

from hail import MatrixTable
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
    Env.hc()._backend.interpret(MatrixMultiWrite([mt._mir for mt in mts], writer))

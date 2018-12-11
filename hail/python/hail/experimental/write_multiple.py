from typing import List

from hail import MatrixTable
from hail.typecheck import sequenceof, typecheck_method

@typecheck_method(objs=sequenceof(MatrixTable),
                  prefix=str,
                  overwrite=bool,
                  stage_locally=bool)
def write_matrix_tables(objs: List[MatrixTable], prefix: str, overwrite: bool = False,
                        stage_locally: bool = False):
    pass  # TODO, implement

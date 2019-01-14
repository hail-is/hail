from hail.ir import BlockMatrixIR
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(path=str)
    def __init__(self, path):
        super().__init__()
        self._path = path

    def render(self, r):
        return f'(BlockMatrixRead "{escape_str(self._path)}")'

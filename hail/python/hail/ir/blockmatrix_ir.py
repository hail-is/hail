from hail.ir import BlockMatrixIR
from hail.utils.java import escape_str
from hail.typecheck import typecheck_method

from hail.utils.java import Env


class BlockMatrixRead(BlockMatrixIR):
    @typecheck_method(path=str)
    def __init__(self, path):
        super().__init__()
        self._path = path

    def render(self, r):
        return f'(BlockMatrixRead "{escape_str(self._path)}")'


class BlockMatrixAdd(BlockMatrixIR):
    @typecheck_method(left=BlockMatrixIR, right=BlockMatrixIR)
    def __init__(self, left, right):
        super().__init__()
        self._left = left
        self._right = right

    def render(self, r):
        return f'(BlockMatrixAdd {r(self._left)} {r(self._right)})'


class JavaBlockMatrix(BlockMatrixIR):
    def __init__(self, jbm):
        super().__init__()
        self._jbm = jbm

    def render(self, r):
        return f'(JavaBlockMatrix {r.add_jir(Env.hail().expr.ir.BlockMatrixLiteral(self._jbm))})'

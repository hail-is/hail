package is.hail.expr.ir

import is.hail.expr._

object Optimize {
  private def optimize(ir0: BaseIR): BaseIR = {
    var ir = ir0
    ir = FoldConstants(ir)
    ir = Simplify(ir)
    ir
  }

  def apply(ir: TableIR): TableIR = optimize(ir).asInstanceOf[TableIR]

  def apply(ir: MatrixIR): MatrixIR = optimize(ir).asInstanceOf[MatrixIR]

  def apply(ir: IR): IR = optimize(ir).asInstanceOf[IR]
}

package is.hail.expr.ir

import is.hail.expr._

object Optimize {
  private def optimize(ir0: BaseIR): BaseIR = {
    var ir = ir0
    ir = FoldConstants(ir)
    ir = Simplify(ir)
    ir = PruneDeadFields(ir)

    assert(ir.typ == ir0.typ, s"optimization changed type!\n  before: ${ir0.typ}\n  after:  ${ir.typ}")
    ir
  }

  def apply(ir: TableIR): TableIR = optimize(ir).asInstanceOf[TableIR]

  def apply(ir: MatrixIR): MatrixIR = optimize(ir).asInstanceOf[MatrixIR]

  def apply(ir: IR): IR = optimize(ir).asInstanceOf[IR]
}

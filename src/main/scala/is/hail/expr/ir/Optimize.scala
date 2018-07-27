package is.hail.expr.ir

import is.hail.utils._

object Optimize {
  private def optimize(ir0: BaseIR, noisy: Boolean): BaseIR = {
    if (noisy)
      log.info("optimize: before:\n" + Pretty(ir0))

    var ir = ir0
    ir = FoldConstants(ir)
    ir = Simplify(ir)
    ir = PruneDeadFields(ir)

    if (ir.typ != ir0.typ)
      fatal(s"optimization changed type!\n  before: ${ir0.typ}\n  after:  ${ir.typ}")

    if (noisy)
      log.info("optimize: after:\n" + Pretty(ir))

    ir
  }

  def apply(ir: TableIR, noisy: Boolean): TableIR = optimize(ir, noisy).asInstanceOf[TableIR]
  def apply(ir: TableIR): TableIR = optimize(ir, noisy = true).asInstanceOf[TableIR]

  def apply(ir: MatrixIR, noisy: Boolean): MatrixIR = optimize(ir, noisy).asInstanceOf[MatrixIR]
  def apply(ir: MatrixIR): MatrixIR = optimize(ir, noisy = true).asInstanceOf[MatrixIR]

  def apply(ir: IR, noisy: Boolean): IR = optimize(ir, noisy).asInstanceOf[IR]
  def apply(ir: IR): IR = optimize(ir, noisy = true).asInstanceOf[IR]
}

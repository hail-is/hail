package is.hail.expr.ir

import is.hail.utils._

object Optimize {
  private def optimize(ir0: BaseIR, noisy: Boolean, canGenerateLiterals: Boolean): BaseIR = {
    if (noisy)
      log.info("optimize: before:\n" + Pretty(ir0))

    var ir = ir0
    ir = FoldConstants(ir, canGenerateLiterals = canGenerateLiterals)
    ir = Simplify(ir)
    ir = PruneDeadFields(ir)

    if (ir.typ != ir0.typ)
      fatal(s"optimization changed type!\n  before: ${ ir0.typ }\n  after:  ${ ir.typ }" +
        s"\n  Before IR:\n  ----------\n${ Pretty(ir0) }\n  After IR:\n  ---------\n${ Pretty(ir) }")

    if (noisy)
      log.info("optimize: after:\n" + Pretty(ir))

    ir
  }

  def apply(ir: TableIR): TableIR = optimize(ir, noisy = true, canGenerateLiterals = true).asInstanceOf[TableIR]

  def apply(ir: MatrixIR): MatrixIR = optimize(ir, noisy = true, canGenerateLiterals = true).asInstanceOf[MatrixIR]

  def apply(ir: IR, noisy: Boolean, canGenerateLiterals: Boolean): IR = optimize(ir, noisy, canGenerateLiterals).asInstanceOf[IR]
  def apply(ir: IR): IR = optimize(ir, noisy = true, canGenerateLiterals = true).asInstanceOf[IR]
}

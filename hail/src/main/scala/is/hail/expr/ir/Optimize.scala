package is.hail.expr.ir

import is.hail.HailContext
import is.hail.utils._

object Optimize {
  private def optimize(ir0: BaseIR, noisy: Boolean, context: Option[String]): BaseIR = {
    val contextStr = context.map(s => s" ($s)").getOrElse("")
    if (noisy)
      log.info(s"optimize$contextStr: before: IR size ${ IRSize(ir0) }: \n" + Pretty(ir0, elideLiterals = true))

    var ir = ir0
    var last: BaseIR = null
    var iter = 0
    val maxIter = HailContext.get.optimizerIterations
    while (iter < maxIter && ir != last) {
      last = ir
      ir = FoldConstants(ir)
      ir = ExtractIntervalFilters(ir)
      ir = Simplify(ir)
      ir = ForwardLets(ir)
      ir = ForwardRelationalLets(ir)
      ir = PruneDeadFields(ir)

      iter += 1
    }

    if (ir.typ != ir0.typ)
      fatal(s"optimization changed type!\n  before: ${ ir0.typ.parsableString() }\n  after:  ${ ir.typ.parsableString() }" +
        s"\n  Before IR:\n  ----------\n${ Pretty(ir0) }\n  After IR:\n  ---------\n${ Pretty(ir) }")

    if (noisy)
      log.info(s"optimize$contextStr: after: IR size ${ IRSize(ir) }:\n" + Pretty(ir, elideLiterals = true))

    ir
  }

  def apply(ir: TableIR, noisy: Boolean): TableIR =
    optimize(ir, noisy, None).asInstanceOf[TableIR]

  def apply(ir: TableIR): TableIR = apply(ir, true)

  def apply(ir: MatrixIR, noisy: Boolean): MatrixIR =
   optimize(ir, noisy, None).asInstanceOf[MatrixIR]

  def apply(ir: MatrixIR): MatrixIR = apply(ir, true)

  def apply(ir: IR, noisy: Boolean, context: Option[String]): IR =
    optimize(ir, noisy, context).asInstanceOf[IR]

  def apply(ir: IR): IR = apply(ir, true, None)
}

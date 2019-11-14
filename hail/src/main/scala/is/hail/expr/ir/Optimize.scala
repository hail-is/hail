package is.hail.expr.ir

import is.hail.HailContext
import is.hail.utils._

object Optimize {
  def optimize(ir0: BaseIR, noisy: Boolean, context: String, ctx: Option[ExecuteContext] = None): BaseIR = {
    if (noisy)
      log.info(s"optimize $context: before: IR size ${ IRSize(ir0) }: \n" + Pretty(ir0, elideLiterals = true))

    def maybeTime[T](x: => T): T = {
      ctx match {
        case Some(ctx) => ctx.timer.time("Optimize")(x)
        case None => x
      }
    }

    var ir = ir0
    var last: BaseIR = null
    var iter = 0
    val maxIter = HailContext.get.optimizerIterations

    def runOpt(f: BaseIR => BaseIR, iter: Int, optContext: String): Unit = {
      ctx match {
        case None =>
          ir = f(ir)
        case Some(ctx) =>
          ir = ctx.timer.time(optContext)(f(ir))
      }
    }

    maybeTime({
      while (iter < maxIter && ir != last) {
        last = ir
        runOpt(FoldConstants(_), iter, "FoldConstants")
        runOpt(ExtractIntervalFilters(_), iter, "ExtractIntervalFilters")
        runOpt(Simplify(_), iter, "Simplify")
        runOpt(ForwardLets(_), iter, "ForwardLets")
        runOpt(ForwardRelationalLets(_), iter, "ForwardRelationalLets")
        runOpt(PruneDeadFields(_), iter, "PruneDeadFields")

        iter += 1
      }
    })

    if (ir.typ != ir0.typ)
      throw new RuntimeException(s"optimization changed type!" +
        s"\n  before: ${ ir0.typ.parsableString() }" +
        s"\n  after:  ${ ir.typ.parsableString() }" +
        s"\n  Before IR:\n  ----------\n${ Pretty(ir0) }" +
        s"\n  After IR:\n  ---------\n${ Pretty(ir) }")

    if (noisy)
      log.info(s"optimize $context: after: IR size ${ IRSize(ir) }:\n" + Pretty(ir, elideLiterals = true))

    ir
  }

  def apply(ir: TableIR, noisy: Boolean): TableIR =
    optimize(ir, noisy, "").asInstanceOf[TableIR]

  def apply(ir: TableIR): TableIR = apply(ir, true)

  def apply(ir: MatrixIR, noisy: Boolean): MatrixIR =
    optimize(ir, noisy, "").asInstanceOf[MatrixIR]

  def apply(ir: MatrixIR): MatrixIR = apply(ir, true)

  def apply(ir: IR, noisy: Boolean, context: String): IR =
    optimize(ir, noisy, context).asInstanceOf[IR]

  def apply(ir: IR): IR = apply(ir, true, "")
}

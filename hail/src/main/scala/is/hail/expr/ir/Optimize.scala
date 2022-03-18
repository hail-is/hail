package is.hail.expr.ir

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.utils._

object Optimize {
  def apply[T <: BaseIR](ir0: T, context: String, ctx: ExecuteContext): T = {
    if (ctx.shouldLogIR())
      log.info(s"optimize $context: before: IR size ${ IRSize(ir0) }: \n" + Pretty(ctx, ir0, elideLiterals = true))

    var ir = ir0
    var last: BaseIR = null
    var iter = 0
    val maxIter = HailContext.get.optimizerIterations

    def runOpt(f: BaseIR => BaseIR, iter: Int, optContext: String): Unit = {
      ir = ctx.timer.time(optContext)(f(ir).asInstanceOf[T])
    }

    ctx.timer.time("Optimize") {
      while (iter < maxIter && ir != last) {
        last = ir
        runOpt(FoldConstants(ctx, _), iter, "FoldConstants")
        runOpt(ExtractIntervalFilters(ctx, _), iter, "ExtractIntervalFilters")
        runOpt(Simplify(ctx, _), iter, "Simplify")
        runOpt(ForwardLets(_), iter, "ForwardLets")
        runOpt(ForwardRelationalLets(_), iter, "ForwardRelationalLets")
        runOpt(PruneDeadFields(ctx, _), iter, "PruneDeadFields")

        iter += 1
      }
    }

    if (ir.typ != ir0.typ)
      throw new RuntimeException(s"optimization changed type!" +
        s"\n  before: ${ ir0.typ.parsableString() }" +
        s"\n  after:  ${ ir.typ.parsableString() }" +
        s"\n  Before IR:\n  ----------\n${ Pretty(ctx, ir0) }" +
        s"\n  After IR:\n  ---------\n${ Pretty(ctx, ir) }")

    if (ctx.shouldLogIR())
      log.info(s"optimize $context: after: IR size ${ IRSize(ir) }:\n" + Pretty(ctx, ir, elideLiterals = true))

    ir
  }
}

package is.hail.expr.ir

import is.hail.HailContext
import is.hail.backend.ExecuteContext

import scala.util.control.Breaks.{break, breakable}

object Optimize {

  private[this] val optimizations: Array[(ExecuteContext, BaseIR) => BaseIR] =
    Array(
      FoldConstants.apply,
      ExtractIntervalFilters.apply,
      NormalizeNames(allowFreeVariables = true),
      Simplify.apply,
      ForwardLets.apply,
      ForwardRelationalLets.apply,
      PruneDeadFields.apply,
    )

  def apply[T <: BaseIR](ctx: ExecuteContext, ir0: T): T =
    ctx.time {

      var ir: BaseIR = ir0

      breakable {
        for (iter <- 0 until HailContext.get.optimizerIterations) {
          val last = ir

          for (f <- optimizations)
            ir = f(ctx, ir)

          TypeCheck(ctx, ir)

          if (ir.typ != last.typ)
            throw new RuntimeException(
              s"Optimize[iteration=$iter] changed type!" +
                s"\n  before: ${last.typ.parsableString()}" +
                s"\n  after:  ${ir.typ.parsableString()}" +
                s"\n  Before IR:\n  ----------\n${Pretty(ctx, last)}" +
                s"\n  After IR:\n  ---------\n${Pretty(ctx, ir)}"
            )

          if (ir == last) break
        }
      }

      ir.asInstanceOf[T]
    }
}

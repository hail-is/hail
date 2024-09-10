package is.hail.expr.ir

import is.hail.HailContext
import is.hail.backend.ExecuteContext
import is.hail.utils.fatal

import scala.util.control.Breaks.{break, breakable}

object Optimize {
  def apply[T <: BaseIR](ctx: ExecuteContext, ir0: T): T = {
    var ir = ir0

    def runOpt(f: BaseIR => BaseIR, iter: Int, optContext: String): Unit =
      ir = ctx.timer.time(s"$optContext, iteration: $iter")(f(ir).asInstanceOf[T])

    breakable {
      for (iter <- 0 until HailContext.get.optimizerIterations) {
        val last = ir
        runOpt(FoldConstants(ctx, _), iter, "FoldConstants")
        runOpt(ExtractIntervalFilters(ctx, _), iter, "ExtractIntervalFilters")
        runOpt(
          NormalizeNames(ctx, _, allowFreeVariables = true),
          iter,
          "NormalizeNames",
        )
        runOpt(Simplify(ctx, _), iter, "Simplify")
        val ircopy = ir.deepCopy()
        runOpt(ForwardLets(ctx), iter, "ForwardLets")
        try
          TypeCheck(ctx, ir)
        catch {
          case e: Exception =>
            fatal(
              s"bad ir from forward lets, started as\n${Pretty(ctx, ircopy, preserveNames = true)}",
              e,
            )
        }
        runOpt(ForwardRelationalLets(_), iter, "ForwardRelationalLets")
        TypeCheck(ctx, ir)
        runOpt(PruneDeadFields(ctx, _), iter, "PruneDeadFields")

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

    ir
  }
}

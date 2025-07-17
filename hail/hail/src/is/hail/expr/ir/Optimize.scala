package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.utils.HailException

import scala.util.control.Breaks.{break, breakable}

object Optimize {
  object Flags {
    val Optimize: String = "optimize"
    val MaxOptimizerIterations: String = "max_optimizer_iterations"
  }

  private[this] val DefaultOptimizerIterations: Int = 3

  private[this] val Optimizations: Array[(ExecuteContext, BaseIR) => BaseIR] =
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

      val iters: Option[Int] =
        ctx.flags.lookup(Flags.MaxOptimizerIterations).map { s =>
          val iters =
            try s.toInt
            catch {
              case _: NumberFormatException =>
                throw new HailException(
                  f"max_optimizer_iterations must be a positive integer, got '$s'."
                )
            }

          if (iters < 0)
            throw new HailException(
              f"max_optimizer_iterations must be greater than 0, got '$iters'."
            )

          iters
        }

      breakable {

        for (iter <- 0 until iters.getOrElse(DefaultOptimizerIterations)) {
          val last = ir

          for (f <- Optimizations)
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

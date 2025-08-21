package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.utils.{fatal, HailException}

import scala.util.control.Breaks.{break, breakable}
import scala.util.control.NonFatal

object Optimize {
  object Flags {
    val MaxOptimizerIterations: String = "max_optimizer_iterations"
    val Optimize: String = "optimize"
  }

  private[this] val DefaultOptimizerIterations: Int = 3

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
                  f"'${Flags.MaxOptimizerIterations}' must be a positive integer, got '$s'."
                )
            }

          if (iters < 0)
            throw new HailException(
              f"'${Flags.MaxOptimizerIterations}' must be greater than 0, got '$iters'."
            )

          iters
        }

      breakable {
        for (iter <- 0 until iters.getOrElse(DefaultOptimizerIterations)) {
          ctx.timer.time(f"iteration $iter") {
            val last = ir

            ir = FoldConstants(ctx, ir)
            ir = ExtractIntervalFilters(ctx, ir)
            ir = NormalizeNames(allowFreeVariables = true)(ctx, ir)
            ir = Simplify(ctx, ir)

            ir = {
              val ir_ = ForwardLets(ctx, ir)
              try
                TypeCheck(ctx, ir_)
              catch {
                case NonFatal(e) =>
                  fatal(
                    s"bad ir from ForwardLets, started as\n${Pretty(ctx, ir, preserveNames = true)}",
                    e,
                  )
              }
              ir_
            }

            ir = ForwardRelationalLets(ctx, ir)
            ir = PruneDeadFields(ctx, ir)

            TypeCheck(ctx, ir)

            if (ir.typ != last.typ)
              throw new RuntimeException(
                s"Optimize[iteration=$iter] changed type!" +
                  s"\n  before: ${last.typ.parsableString()}" +
                  s"\n  after:  ${ir.typ.parsableString()}" +
                  s"\n  Before IR:\n  ----------\n${Pretty(ctx, last)}" +
                  s"\n  After IR:\n  ---------\n${Pretty(ctx, ir)}"
              )

            if (ir == last) break()
          }
        }
      }

      ir.asInstanceOf[T]
    }
}

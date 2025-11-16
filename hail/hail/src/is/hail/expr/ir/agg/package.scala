package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.utils.HailException

package agg {
  object Flags {
    val BranchingFactor = "branching_factor"
  }
}

package object agg {
  private[this] val DefaultBranchingFactor: Int = 50

  implicit class AggExecuteContextExtensions(private val ctx: ExecuteContext) extends AnyVal {
    def branchingFactor: Int =
      ctx.flags
        .lookup(Flags.BranchingFactor)
        .map { s =>
          val factor =
            try s.toInt
            catch {
              case _: NumberFormatException =>
                throw new HailException(
                  f"'${Flags.BranchingFactor}' must be a positive integer, got '$s'."
                )
            }

          if (factor < 0)
            throw new HailException(
              f"'${Flags.BranchingFactor}' must be greater than 0, got '$factor'."
            )

          factor
        }
        .getOrElse(DefaultBranchingFactor)
  }

}

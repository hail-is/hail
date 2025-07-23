package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.utils.HailException

package agg {
  object Flags {
    val BranchFactor = "branch_factor"
  }
}

package object agg {
  val DefaultBranchFactor: Int = 50

  def branchFactor(ctx: ExecuteContext): Int = {
    val factor =
      ctx.flags.lookup(Flags.BranchFactor).map { s =>
        val factor =
          try s.toInt
          catch {
            case _: NumberFormatException =>
              throw new HailException(
                f"'${Flags.BranchFactor}' must be a positive integer, got '$s'."
              )
          }

        if (factor < 0)
          throw new HailException(
            f"'${Flags.BranchFactor}' must be greater than 0, got '$factor'."
          )

        factor
      }

    factor.getOrElse(DefaultBranchFactor)
  }
}

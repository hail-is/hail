package is.hail.expr.ir.implicits

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, NormalizeNames}

object forTesting {
  implicit class BaseIROps(private val ir: BaseIR) extends AnyVal {
    def isAlphaEquiv(ctx: ExecuteContext, other: BaseIR): Boolean = {
      // FIXME: rewrite to not rebuild the irs by maintaining an env mapping left to right names
      val normalize: (ExecuteContext, BaseIR) => BaseIR = NormalizeNames(allowFreeVariables = true)
      normalize(ctx, ir) == normalize(ctx, other)
    }
  }
}

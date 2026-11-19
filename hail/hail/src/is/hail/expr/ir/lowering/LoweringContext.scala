package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, Requiredness, RequirednessAnalysis}

object LoweringContext {
  def apply(ctx: ExecuteContext, root: BaseIR): LoweringContext =
    new LoweringContext(ctx, root)
}

final class LoweringContext private (val ctx: ExecuteContext, val root: BaseIR) {
  lazy val requiredness: RequirednessAnalysis = Requiredness(root, ctx)
  lazy val distinctKeyed: DistinctKeyedAnalysis = DistinctlyKeyed(root)
}

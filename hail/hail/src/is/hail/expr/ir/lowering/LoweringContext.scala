package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, Requiredness, RequirednessAnalysis}
import is.hail.utils.{lazily, Lazy}

object LoweringContext {
  def apply(ctx: ExecuteContext, root: BaseIR): LoweringContext =
    new LoweringContext(
      ctx,
      root,
      lazily(Requiredness(root, ctx)),
      lazily(DistinctlyKeyed(root)),
    )
}

final class LoweringContext(
  val ctx: ExecuteContext,
  val root: BaseIR,
  val requiredness: Lazy[RequirednessAnalysis],
  val distinctKeyed: Lazy[DistinctKeyedAnalysis],
)

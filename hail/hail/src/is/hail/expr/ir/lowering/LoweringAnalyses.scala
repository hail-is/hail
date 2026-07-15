package is.hail.expr.ir.lowering

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{BaseIR, Requiredness, RequirednessAnalysis}

object LoweringAnalyses {
  def apply(ir: BaseIR, ctx: ExecuteContext): LoweringAnalyses = {
    val requirednessAnalysis = Requiredness(ir, ctx)
    val distinctKeyedAnalysis = DistinctlyKeyed.apply(ir)
    LoweringAnalyses(requirednessAnalysis, distinctKeyedAnalysis)
  }
}

case class LoweringAnalyses(
  requirednessAnalysis: RequirednessAnalysis,
  distinctKeyedAnalysis: DistinctKeyedAnalysis,
)

package is.hail.expr.ir

import is.hail.backend.ExecuteContext

object Analyses {
    def apply(ir: BaseIR, ctx:ExecuteContext): Analyses = {
    val requirednessAnalysis = Requiredness(ir, ctx)
    val distinctKeyedAnalysis = DistinctlyKeyed.apply(ir)
    Analyses(requirednessAnalysis, distinctKeyedAnalysis)
  }
}
case class Analyses(requirednessAnalysis: RequirednessAnalysis, distinctKeyedAnalysis: DistinctKeyedAnalysis)

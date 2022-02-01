package is.hail.expr.ir

import is.hail.backend.ExecuteContext

case class Analyses(ir: BaseIR, ctx:ExecuteContext) {
  val requirednessAnalysis = Requiredness(ir, ctx)
  val distinctKeyedAnalysis = DistinctlyKeyed.apply(ir)
}

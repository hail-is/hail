package is.hail.expr.ir

import is.hail.expr.ir.lowering.MonadLower

import scala.language.higherKinds

object LoweringAnalyses {
    def apply[M[_]](ir: BaseIR)(implicit M: MonadLower[M]): M[LoweringAnalyses] =
      M.ctx.reader { ctx =>
        val requirednessAnalysis = Requiredness(ir, ctx)
        val distinctKeyedAnalysis = DistinctlyKeyed.apply(ir)
        LoweringAnalyses(requirednessAnalysis, distinctKeyedAnalysis)
      }
}
case class LoweringAnalyses(requirednessAnalysis: RequirednessAnalysis,
                            distinctKeyedAnalysis: DistinctKeyedAnalysis
                           )

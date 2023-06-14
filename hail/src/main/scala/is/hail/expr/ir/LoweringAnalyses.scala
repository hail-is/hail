package is.hail.expr.ir

import cats.Functor
import cats.implicits.toFunctorOps
import cats.mtl.Ask
import is.hail.backend.ExecuteContext

import scala.language.higherKinds

object LoweringAnalyses {
  def apply[M[_]: Functor](ir: BaseIR)(implicit A: Ask[M, ExecuteContext]): M[LoweringAnalyses] =
    Requiredness(ir).map(LoweringAnalyses(_, DistinctlyKeyed(ir)))
}

case class LoweringAnalyses(requirednessAnalysis: RequirednessAnalysis,
                            distinctKeyedAnalysis: DistinctKeyedAnalysis
                           )

package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.expr.ir.lowering.{LowerMatrixToTablePass, LoweringPipeline, OptimizePass}

import org.testng.annotations.Test

class LoweringPipelineSuite extends HailSuite {
  @Test def testNoOpt(): Unit = {
    val lp = LoweringPipeline(LowerMatrixToTablePass, OptimizePass("foo"), OptimizePass("bar"))
    assert(lp.noOptimization() == LoweringPipeline(LowerMatrixToTablePass))
  }
}

package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.TestUtils.eval
import is.hail.backend.ExecuteContext
import is.hail.expr.ir
import is.hail.expr.ir.lowering.Lower.monadLowerInstanceForLower
import is.hail.expr.ir.lowering.LoweringState
import is.hail.types.virtual.{TArray, TBoolean, TSet, TString}
import is.hail.utils._
import org.testng.annotations.Test

class MemoryLeakSuite extends HailSuite {
  @Test def testLiteralSetContains(): Unit = {

    val litSize = 32000

   def run(size: Int): Long = {
      val lit = Literal(TSet(TString), (0 until litSize).map(_.toString).toSet)
      val queries = Literal(TArray(TString), (0 until size).map(_.toString).toFastIndexedSeq)
      ExecuteContext.scoped() { ctx =>
        eval(
          ToArray(
            mapIR(ToStream(queries)) { r => ir.invoke("contains", TBoolean, lit, r) }
          ), Env.empty, FastIndexedSeq(), None, None, false, ctx
        ).runA(ctx, LoweringState())
        ctx.r.pool.getHighestTotalUsage
      }
    }

    val size1 = 10
    val size2 = 100
    val mem1 = run(size1)
    val mem2 = run(size2)
    if (mem2 > (mem1 * 1.1))
      throw new AssertionError(s"literal set contains is scaling with number of queries!" +
        s"\n  Memory used with size1=$size1: ${formatSpace(mem1)}" +
        s"\n  Memory used with size1=$size2: ${formatSpace(mem2)}")
  }
}

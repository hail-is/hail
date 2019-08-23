package is.hail.expr.ir

import is.hail.expr.types.virtual.{TFloat64, TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class FoldConstantsSuite extends TestNGSuite {
  @Test def testLiteralGeneration() {
    val x = MakeTuple.ordered(Seq(I32(1)))
    assert(FoldConstants(x) == Literal(TTuple(TInt32()), Row(1)))
    assert(FoldConstants(x, canGenerateLiterals = false) == x)
  }

  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64())
    assert(FoldConstants(x) == x)
  }
}

package is.hail.expr.ir

import is.hail.expr.types.{TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class FoldConstantsSuite extends TestNGSuite {
  @Test def test_let_evaluation() {
    val ir = Let(
      "x",
      I32(5),
      Let(
        "y",
        ApplyBinaryPrimOp(Add(), Ref("x", TInt32()), I32(10)),
        ApplyBinaryPrimOp(Add(), Ref("y", TInt32()), I32(15))
      )
    )
    assert(FoldConstants(ir) == I32(30))
  }

  @Test def testLiteralGeneration() {
    val x = MakeTuple(Seq(I32(1)))
    assert(FoldConstants(x) == Literal(TTuple(TInt32()), Row(1)))
    assert(FoldConstants(x, canGenerateLiterals = false) == x)
  }

  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L)
    assert(FoldConstants(x) == x)
  }
}

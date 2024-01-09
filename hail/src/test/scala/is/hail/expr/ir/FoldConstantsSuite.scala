package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.types.virtual.{TFloat64, TInt32, TRNGState, TTuple}

import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class FoldConstantsSuite extends HailSuite {
  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), RNGStateLiteral(), 0L, TFloat64)
    assert(FoldConstants(ctx, x) == x)
  }

  @Test def testErrorCatching() {
    val ir = invoke("toInt32", TInt32, Str(""))
    assert(FoldConstants(ctx, ir) == ir)
  }

  @DataProvider(name = "aggNodes")
  def aggNodes(): Array[Array[Any]] = {
    Array[IR](
      AggLet("x", I32(1), I32(1), false),
      AggLet("x", I32(1), I32(1), true),
      ApplyAggOp(Sum())(I64(1)),
      ApplyScanOp(Sum())(I64(1)),
    ).map(x => Array[Any](x))
  }

  @Test def testAggNodesConstruction(): Unit = aggNodes()

  @Test(dataProvider = "aggNodes") def testAggNodesDoNotFold(node: IR): Unit =
    assert(FoldConstants(ctx, node) == node)
}

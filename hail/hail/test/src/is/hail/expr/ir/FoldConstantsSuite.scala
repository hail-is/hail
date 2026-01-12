package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{
  AggLet, Apply, ApplyAggOp, ApplyScanOp, F64, I32, I64, RNGSplitStatic, RNGStateLiteral, Str,
}
import is.hail.types.virtual.{TFloat64, TInt32}

import org.testng.annotations.{DataProvider, Test}

class FoldConstantsSuite extends HailSuite {
  @Test def testRandomBlocksFolding(): Unit = {
    val x = Apply(
      "rand_norm",
      FastSeq.empty,
      FastSeq(RNGSplitStatic(RNGStateLiteral(), 0L), F64(0d), F64(0d)),
      TFloat64,
    )
    assert(FoldConstants(ctx, x) == x)
  }

  @Test def testErrorCatching(): Unit = {
    val ir = invoke("toInt32", TInt32, Str(""))
    assert(FoldConstants(ctx, ir) == ir)
  }

  @DataProvider(name = "aggNodes")
  def aggNodes(): Array[Array[Any]] = {
    Array[IR](
      AggLet(freshName(), I32(1), I32(1), false),
      AggLet(freshName(), I32(1), I32(1), true),
      ApplyAggOp(Sum())(I64(1)),
      ApplyScanOp(Sum())(I64(1)),
    ).map(x => Array[Any](x))
  }

  @Test def testAggNodesConstruction(): Unit = aggNodes(): Unit

  @Test(dataProvider = "aggNodes") def testAggNodesDoNotFold(node: IR): Unit =
    assert(FoldConstants(ctx, node) == node)
}

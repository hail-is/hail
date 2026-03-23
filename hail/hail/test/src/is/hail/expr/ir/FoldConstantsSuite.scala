package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{
  AggLet, Apply, ApplyAggOp, ApplyScanOp, F64, I32, I64, RNGSplitStatic, RNGStateLiteral, Str,
}
import is.hail.types.virtual.{TFloat64, TInt32}

class FoldConstantsSuite extends HailSuite {
  test("RandomBlocksFolding") {
    val x = Apply(
      "rand_norm",
      FastSeq.empty,
      FastSeq(RNGSplitStatic(RNGStateLiteral(), 0L), F64(0d), F64(0d)),
      TFloat64,
    )
    assertEquals(FoldConstants(ctx, x), x)
  }

  test("ErrorCatching") {
    val ir = invoke("toInt32", TInt32, Str(""))
    assertEquals(FoldConstants(ctx, ir), ir)
  }

  object aggNodes extends TestCases {
    def apply(node: IR)(implicit loc: munit.Location): Unit =
      test("AggNodesDoNotFold") {
        assertEquals(FoldConstants(ctx, node), node)
      }
  }

  Array[IR](
    AggLet(freshName(), I32(1), I32(1), false),
    AggLet(freshName(), I32(1), I32(1), true),
    ApplyAggOp(Sum())(I64(1)),
    ApplyScanOp(Sum())(I64(1)),
  ).foreach(aggNodes(_))
}

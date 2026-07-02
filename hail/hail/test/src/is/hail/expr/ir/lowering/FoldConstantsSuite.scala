package is.hail.expr.ir.lowering

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.{freshName, invoke, IR, Sum}
import is.hail.expr.ir.defs._
import is.hail.types.virtual.{TFloat64, TInt32}

import org.junit.jupiter.api.Test

class FoldConstantsSuite {
  @Test def testRandomBlocksFolding(implicit ctx: ExecuteContext): Unit = {
    val x = Apply(
      "rand_norm",
      ArraySeq.empty,
      ArraySeq(RNGSplitStatic(RNGStateLiteral(), 0L), F64(0d), F64(0d)),
      TFloat64,
    )
    assertEq(FoldConstants(ctx, x), x)
  }

  @Test def testErrorCatching(implicit ctx: ExecuteContext): Unit = {
    val ir = invoke("toInt32", TInt32, Str(""))
    assertEq(FoldConstants(ctx, ir), ir)
  }

  def testAggNodesDoNotFold() = ArraySeq(
    AggLet(freshName(), I32(1), I32(1), false),
    AggLet(freshName(), I32(1), I32(1), true),
    ApplyAggOp(Sum())(I64(1)),
    ApplyScanOp(Sum())(I64(1)),
  )

  @ParameterizedTest
  def testAggNodesDoNotFold(node: IR)(implicit ctx: ExecuteContext): Unit =
    assertEq(FoldConstants(ctx, node), node)
}

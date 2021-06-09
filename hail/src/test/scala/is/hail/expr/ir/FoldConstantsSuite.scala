package is.hail.expr.ir

import is.hail.HailSuite
import is.hail.types.virtual.{TFloat64, TInt32, TTuple}
import org.apache.spark.sql.Row
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class FoldConstantsSuite extends HailSuite {
  @Test def testRandomBlocksFolding() {
    val x = ApplySeeded("rand_norm", Seq(F64(0d), F64(0d)), 0L, TFloat64)
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
      ApplyScanOp(Sum())(I64(1))
      ).map(x => Array[Any](x))
  }

  @Test def testAggNodesConstruction(): Unit = aggNodes()

  @Test(dataProvider = "aggNodes") def testAggNodesDoNotFold(node: IR): Unit = {
    assert(FoldConstants(ctx, node) == node)
  }

  @Test def testFindConstantSubtrees(): Unit = {
    val i4 = I32(4)
    val i8 = I32(8)
    val ref = "x"
    val range = StreamRange(0, 10, 1)

    val oppIR = ApplyBinaryPrimOp(Add(), i4, i8)
    val refOppIR = ApplyBinaryPrimOp(Add(), Ref(ref, TInt32), i8)
    val letIR = Let(ref, i4, refOppIR)
    val streamMapIR = StreamMap(range, ref, refOppIR)


    val oppTest = FoldConstants.findConstantSubTrees(oppIR)
    assert(oppTest.contains(oppIR))

    val letTest = FoldConstants.findConstantSubTrees(letIR)
    assert(letTest.contains(letIR))

    val streamMapTest = FoldConstants.findConstantSubTrees(streamMapIR)
    assert(streamMapTest.contains(streamMapIR))
  }
}

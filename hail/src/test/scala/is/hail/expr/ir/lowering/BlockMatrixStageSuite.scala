package is.hail.expr.ir.lowering

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.TestUtils._
import is.hail.types.BlockMatrixSparsity
import is.hail.utils._
import org.testng.annotations.Test

class BlockMatrixStageSuite extends HailSuite {

  private[this] implicit val execStrats: Set[ExecStrategy.ExecStrategy] = ExecStrategy.compileOnly

  def collected(
    ctxs: Array[((Int, Int), IR)],
    globalVals: Array[(String, IR)],
    body: Ref => IR = ref => ref,
    order: Option[Array[(Int, Int)]] = None
  ): IR = {
    val stage = if (ctxs.isEmpty)
      BlockMatrixStage.empty(TInt32)
    else {
      new BlockMatrixStage(
        globalVals,
        ctxs.head._2.typ) {
        private[this] val ctxMap = ctxs.toMap
        def blockContext(idx: (Int, Int)): IR = ctxMap(idx)
        def blockBody(ctxRef: Ref): IR = body(ctxRef)
      }
    }
    stage.collectBlocks(Seq())(b => b, order.getOrElse(ctxs.map(_._1)))
  }

  @Test def testBlockMatrixCollectOrdering(): Unit = {
    val ctxs = Array.tabulate[((Int, Int), IR)](5) { i => ((i, 6-i), In(0, TInt32) + I32(i)) }
    assertEvalsTo(
      collected(ctxs, Array(), order = Some(Array.tabulate(5)(i => (4-i, i + 2)))),
      args = IndexedSeq(0 -> TInt32),
      expected = IndexedSeq(4, 3, 2, 1, 0))
  }

  @Test def testContextDependsOnGlobalValue(): Unit = {
    val g1 = "x" -> In(0, TString)
    assertEvalsTo(
      collected(Array.tabulate(5)(i => (i -> i, Ref("x", TString))),
        Array(g1)),
      args = IndexedSeq("foo" -> TString),
      expected = Array.fill(5)("foo").toFastIndexedSeq)
  }

  @Test def testBodyDependsOnGlobalValue(): Unit = {
    val g1 = "x" -> In(0, TInt32)
    assertEvalsTo(
      ToSet(ToStream(
        collected(Array.tabulate(5)(i => (i -> i, I32(i))),
          Array(g1),
          body=ref => ref + Ref("x", TInt32)))),
      args = IndexedSeq(5 -> TInt32),
      expected = Array.tabulate(5)(i => i + 5).toSet)
  }

  @Test def testGlobalValueDependentBinding(): Unit = {
    val g1 = "x" -> In(0, TString)
    val g2 = "y" -> MakeArray(FastIndexedSeq(Ref("x", TString)), TArray(TString))

    assertEvalsTo(
      collected(Array.tabulate(5)(i => (i -> i, Ref("y", TArray(TString)))),
        Array(g1, g2)),
      args = IndexedSeq("foo" -> TString),
      expected = Array.fill(5)(IndexedSeq("foo")).toFastIndexedSeq)
  }

  @Test def testEmptyContexts(): Unit = {
    assertEvalsTo(
      collected(Array(),
        Array(),
        body = ref => ref + I32(5)),
      expected = IndexedSeq())
  }

}

package is.hail.expr.ir

import is.hail.expr.types.virtual.{TInt32, TVoid}
import is.hail.TestUtils._
import is.hail.expr.ir
import is.hail.utils.FastSeq
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class ForwardLetsSuite extends TestNGSuite {
  @DataProvider(name="nonForwardingOps")
  def nonForwardingOps(): Array[Array[IR]] = {
    val a = ArrayRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt32())
    Array(
      ArrayMap(a, "y", ApplyBinaryPrimOp(Add(), x, y)),
      ArrayFilter(a, "y", ApplyComparisonOp(LT(TInt32()),x, y)),
      ArrayFlatMap(a, "y", ArrayRange(x, y, I32(1))),
      ArrayFold(a, I32(0), "acc", "y", ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), Ref("acc", TInt32()))),
      ArrayScan(a, I32(0), "acc", "y", ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, y), Ref("acc", TInt32()))),
      MakeStruct(FastSeq("a" -> ApplyBinaryPrimOp(Add(), x, I32(1)), "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)))),
      MakeTuple(FastSeq(ApplyBinaryPrimOp(Add(), x, I32(1)), ApplyBinaryPrimOp(Add(), x, I32(2)))),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), x, x), I32(1))
    ).map(ir => Array[IR](Let("x", In(0, TInt32()), ir)))
  }

  @DataProvider(name="forwardingOps")
  def forwardingOps(): Array[Array[IR]] = {
    val a = ArrayRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    Array(
      MakeStruct(FastSeq("a" -> I32(1), "b" -> ApplyBinaryPrimOp(Add(), x, I32(2)))),
      MakeTuple(FastSeq(I32(1), ApplyBinaryPrimOp(Add(), x, I32(2)))),
      If(True(), x, I32(0)),
      ApplyBinaryPrimOp(Add(), ApplyBinaryPrimOp(Add(), I32(2), x), I32(1)),
      ApplyUnaryPrimOp(Negate(), x)
    ).map(ir => Array[IR](Let("x", In(0, TInt32()), ir)))
  }

  @Test def assertDataProvidersWork() {
    nonForwardingOps()
    forwardingOps()
  }

  @Test(dataProvider = "nonForwardingOps")
  def testNonForwardingOps(ir: IR): Unit = {
    val after = ForwardLets(ir)
    assert(after.isInstanceOf[Let])
    assertEvalSame(ir, args = Array(5 -> TInt32()))
  }

  @Test(dataProvider = "forwardingOps")
  def testForwardingOps(ir: IR): Unit = {
   val after = ForwardLets(ir)
    assert(!after.isInstanceOf[Let])
    assertEvalSame(ir, args = Array(5 -> TInt32()))
  }

  @Test def testLetNoMention(): Unit = {
    val ir = Let("x", I32(1), I32(2))
    assert(ForwardLets(ir) == I32(2))
  }

  @Test def testLetRefRewrite(): Unit = {
    val ir = Let("x", I32(1), Ref("x", TInt32()))
    assert(ForwardLets(ir) == I32(1))
  }
}

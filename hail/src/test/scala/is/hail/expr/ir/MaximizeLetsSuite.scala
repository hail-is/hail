package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.types.virtual._
import is.hail.utils.FastSeq
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

class MaximizeLetsSuite extends TestNGSuite {
  @DataProvider(name = "nonMaximizingOps")
  def nonMaximizingOps(): Array[Array[IR]] = {
    val a = ArrayRange(I32(0), I32(10), I32(1))
    val x = Ref("x", TInt32())
    val y = Ref("y", TInt32())

    def let(t: Type, name: String = "x") = Let(name, ApplyBinaryPrimOp(Add(), I32(1), y), NA(t))

    Array(
      ArrayMap(a, "y", let(TInt32())),
      ArrayFilter(a, "y", let(TBoolean())),
      ArrayFlatMap(a, "y", let(TArray(TInt32()))),
      ArrayFold(a, I32(0), "acc", "y", let(TInt32())),
      ArrayFold(a, I32(0), "acc", "y", let(TInt32(), "acc")),
      ArrayScan(a, I32(0), "acc", "y", let(TInt32())),
      ArrayScan(a, I32(0), "acc", "y", let(TInt32(), "acc"))
    ).map(ir => Array[IR](ir))
  }

  @DataProvider(name = "maximizingOps")
  def maximizingOps(): Array[Array[IR]] = {
    val x = Ref("x", TInt32())
    val l = Let("x", I32(1), ApplyBinaryPrimOp(Add(), x, x))
    Array(
      MakeStruct(FastSeq("a" -> l)),
      MakeTuple(FastSeq(l)),
      ApplyBinaryPrimOp(Add(), l, I32(2)),
      ApplyUnaryPrimOp(Negate(), l),
      If(True(), l, NA(TInt32()))).map(ir => Array[IR](ir))
  }

  @Test def assertDataProvidersWork(): Unit = {
    nonMaximizingOps()
    maximizingOps()
  }

  @Test(dataProvider = "nonMaximizingOps")
  def testNonMaximizingOps(ir: IR): Unit = {
    val after = MaximizeLets(ir)
    assert(!after.isInstanceOf[Let])
    TypeCheck(ir)
  }

  @Test(dataProvider = "maximizingOps")
  def testMaximizingOps(ir: IR): Unit = {
    val after = MaximizeLets(ir)
    assert(after.isInstanceOf[Let])
    TypeCheck(ir)
  }

  @Test def testNestedLets(): Unit = {
    val ir = Let(
      "x1",
      I32(1),
      Let(
        "x2",
        I32(2),
        ApplyUnaryPrimOp(Negate(), Ref("x2", TInt32()))
      )
    )

    assert(MaximizeLets(ir) == ir)
  }

  @Test def testEquivalentLets(): Unit = {
    val ir = Let(
      "x1",
      I32(1),
      Let(
        "toElide",
        I32(1),
        ApplyUnaryPrimOp(Negate(), Ref("toElide", TInt32()))
      )
    )

    assert(MaximizeLets(ir) == Let("x1", I32(1), ApplyUnaryPrimOp(Negate(), Ref("x1", TInt32()))))
  }
}

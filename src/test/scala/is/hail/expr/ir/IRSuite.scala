package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class IRSuite extends TestNGSuite {

  @Test def testI32() {
    assertEvalsTo(I32(5), 5)
  }

  @Test def testI64() {
    assertEvalsTo(I64(5), 5L)
  }

  @Test def testF32() {
    assertEvalsTo(F32(3.14f), 3.14f)
  }

  @Test def testF64() {
    assertEvalsTo(F64(3.14), 3.14)
  }

  @Test def testStr() {
    assertEvalsTo(Str("Hail"), "Hail")
  }

  @Test def testTrue() {
    assertEvalsTo(True(), true)
  }

  @Test def testFalse() {
    assertEvalsTo(False(), false)
  }

  // FIXME Void() doesn't work becuase we can't handle a void type in a tuple

  @Test def testCast() {
    assertEvalsTo(Cast(I32(5), TInt32()), 5)
    assertEvalsTo(Cast(I32(5), TInt64()), 5L)
    assertEvalsTo(Cast(I32(5), TFloat32()), 5.0f)
    assertEvalsTo(Cast(I32(5), TFloat64()), 5.0)

    assertEvalsTo(Cast(I64(5), TInt32()), 5)
    assertEvalsTo(Cast(I64(0xf29fb5c9af12107dL), TInt32()), 0xaf12107d) // truncate
    assertEvalsTo(Cast(I64(5), TInt64()), 5L)
    assertEvalsTo(Cast(I64(5), TFloat32()), 5.0f)
    assertEvalsTo(Cast(I64(5), TFloat64()), 5.0)

    assertEvalsTo(Cast(F32(3.14f), TInt32()), 3)
    assertEvalsTo(Cast(F32(3.99f), TInt32()), 3) // truncate
    assertEvalsTo(Cast(F32(3.14f), TInt64()), 3L)
    assertEvalsTo(Cast(F32(3.14f), TFloat32()), 3.14f)
    assertEvalsTo(Cast(F32(3.14f), TFloat64()), 3.14)

    assertEvalsTo(Cast(F64(3.14), TInt32()), 3)
    assertEvalsTo(Cast(F64(3.99), TInt32()), 3) // truncate
    assertEvalsTo(Cast(F64(3.14), TInt64()), 3L)
    assertEvalsTo(Cast(F64(3.14), TFloat32()), 3.14f)
    assertEvalsTo(Cast(F64(3.14), TFloat64()), 3.14)
  }

  @Test def testNA() {
    assertEvalsTo(NA(TInt32()), null)
  }

  @Test def testIsNA() {
    assertEvalsTo(IsNA(NA(TInt32())), true)
    assertEvalsTo(IsNA(I32(5)), false)
  }

  @Test def testIf() {
    assertEvalsTo(If(True(), I32(5), I32(7)), 5)
    assertEvalsTo(If(False(), I32(5), I32(7)), 7)
    assertEvalsTo(If(NA(TBoolean()), I32(5), I32(7)), null)
    assertEvalsTo(If(True(), NA(TInt32()), I32(7)), null)
  }

  @Test def testLet() {
    assertEvalsTo(Let("v", I32(5), Ref("v", TInt32())), 5)
    assertEvalsTo(Let("v", NA(TInt32()), Ref("v", TInt32())), null)
    assertEvalsTo(Let("v", I32(5), NA(TInt32())), null)
  }

  @Test def testMakeArray() {
    assertEvalsTo(MakeArray(FastSeq(I32(5), NA(TInt32()), I32(-3)), TArray(TInt32())), FastIndexedSeq(5, null, -3))
    assertEvalsTo(MakeArray(FastSeq(), TArray(TInt32())), FastIndexedSeq())
  }

  @Test def testArrayRef() {
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(0)), 5)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(1)), null)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), NA(TInt32())), null)
  }

  @Test def testArrayLen() {
    assertEvalsTo(ArrayLen(NA(TArray(TInt32()))), null)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(), TArray(TInt32()))), 0)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32()))), 2)
  }

  @Test def testArrayFilter() {
    val naa = NA(TArray(TInt32()))
    val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7)), TArray(TInt32()))

    assertEvalsTo(ArrayFilter(naa, "x", True()), null)

    assertEvalsTo(ArrayFilter(a, "x", NA(TBoolean())), FastIndexedSeq())
    assertEvalsTo(ArrayFilter(a, "x", False()), FastIndexedSeq())
    assertEvalsTo(ArrayFilter(a, "x", True()), FastIndexedSeq(3, null, 7))

    assertEvalsTo(ArrayFilter(a, "x",
      IsNA(Ref("x", TInt32()))), FastIndexedSeq(null))
    assertEvalsTo(ArrayFilter(a, "x",
      ApplyUnaryPrimOp(Bang(), IsNA(Ref("x", TInt32())))), FastIndexedSeq(3, 7))

    assertEvalsTo(ArrayFilter(a, "x",
      ApplyComparisonOp(LT(TInt32()), Ref("x", TInt32()), I32(6))), FastIndexedSeq(3))
  }
}

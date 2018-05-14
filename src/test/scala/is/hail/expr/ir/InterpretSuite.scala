package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.TestUtils._
import org.apache.spark.sql.Row
import org.junit.Test

class InterpretSuite {

  private val env = Env.empty[(Any, Type)]
    .bind("a", (5, TInt32()))

  private val i32 = I32(1)
  private val i32Zero = I32(0)
  private val i64 = I64(2)
  private val i64Zero = I64(0)
  private val f32 = F32(1.1f)
  private val f32Zero = F32(0)
  private val f64 = F64(2.5)
  private val f64Zero = F64(0)
  private val t = True()
  private val f = False()

  private val arr = MakeArray(List(I32(1), I32(5), I32(2), NA(TInt32())), TArray(TInt32()))

  private val struct = MakeStruct(List("a" -> i32, "b" -> f32, "c" -> ArrayRange(I32(0), I32(5), I32(1))))

  private val tuple = MakeTuple(List(i32, f32, ArrayRange(I32(0), I32(5), I32(1))))

  @Test def testUnaryPrimOp() {
    eval(t)
    eval(f)
    eval(ApplyUnaryPrimOp(Bang(), t))
    eval(ApplyUnaryPrimOp(Bang(), f))

    eval(ApplyUnaryPrimOp(Negate(), i32))
    eval(ApplyUnaryPrimOp(Negate(), i64))
    eval(ApplyUnaryPrimOp(Negate(), f32))
    eval(ApplyUnaryPrimOp(Negate(), f64))
  }

  @Test def testApplyBinaryPrimOp() {
    eval(i32)
    eval(i64)
    eval(f32)
    eval(f64)

    eval(ApplyBinaryPrimOp(Add(), i32, i32))
    eval(ApplyBinaryPrimOp(Add(), i64, i64))
    eval(ApplyBinaryPrimOp(Add(), f32, f32))
    eval(ApplyBinaryPrimOp(Add(), f64, f64))

    eval(ApplyBinaryPrimOp(Subtract(), i32, i32))
    eval(ApplyBinaryPrimOp(Subtract(), i64, i64))
    eval(ApplyBinaryPrimOp(Subtract(), f32, f32))
    eval(ApplyBinaryPrimOp(Subtract(), f64, f64))

    eval(ApplyBinaryPrimOp(FloatingPointDivide(), i32, i32))
    eval(ApplyBinaryPrimOp(FloatingPointDivide(), i64, i64))
    eval(ApplyBinaryPrimOp(FloatingPointDivide(), f32, f32))
    eval(ApplyBinaryPrimOp(FloatingPointDivide(), f64, f64))

    eval(ApplyBinaryPrimOp(RoundToNegInfDivide(), i32, i32))
    eval(ApplyBinaryPrimOp(RoundToNegInfDivide(), i64, i64))
    eval(ApplyBinaryPrimOp(RoundToNegInfDivide(), f32, f32))
    eval(ApplyBinaryPrimOp(RoundToNegInfDivide(), f64, f64))

    eval(ApplyBinaryPrimOp(Multiply(), i32, i32))
    eval(ApplyBinaryPrimOp(Multiply(), i64, i64))
    eval(ApplyBinaryPrimOp(Multiply(), f32, f32))
    eval(ApplyBinaryPrimOp(Multiply(), f64, f64))

    eval(ApplyBinaryPrimOp(EQ(), i32, i32))
    eval(ApplyBinaryPrimOp(EQ(), i64, i64))
    eval(ApplyBinaryPrimOp(EQ(), f32, f32))
    eval(ApplyBinaryPrimOp(EQ(), f64, f64))

    eval(ApplyBinaryPrimOp(GT(), i32, i32))
    eval(ApplyBinaryPrimOp(GT(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(GT(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(GT(), i64, i64))
    eval(ApplyBinaryPrimOp(GT(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(GT(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(GT(), f32, f32))
    eval(ApplyBinaryPrimOp(GT(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(GT(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(GT(), f64, f64))
    eval(ApplyBinaryPrimOp(GT(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(GT(), f64, f64Zero))

    eval(ApplyBinaryPrimOp(GTEQ(), i32, i32))
    eval(ApplyBinaryPrimOp(GTEQ(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(GTEQ(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(GTEQ(), i64, i64))
    eval(ApplyBinaryPrimOp(GTEQ(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(GTEQ(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(GTEQ(), f32, f32))
    eval(ApplyBinaryPrimOp(GTEQ(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(GTEQ(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(GTEQ(), f64, f64))
    eval(ApplyBinaryPrimOp(GTEQ(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(GTEQ(), f64, f64Zero))

    eval(ApplyBinaryPrimOp(LT(), i32, i32))
    eval(ApplyBinaryPrimOp(LT(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(LT(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(LT(), i64, i64))
    eval(ApplyBinaryPrimOp(LT(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(LT(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(LT(), f32, f32))
    eval(ApplyBinaryPrimOp(LT(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(LT(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(LT(), f64, f64))
    eval(ApplyBinaryPrimOp(LT(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(LT(), f64, f64Zero))

    eval(ApplyBinaryPrimOp(LTEQ(), i32, i32))
    eval(ApplyBinaryPrimOp(LTEQ(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(LTEQ(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(LTEQ(), i64, i64))
    eval(ApplyBinaryPrimOp(LTEQ(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(LTEQ(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(LTEQ(), f32, f32))
    eval(ApplyBinaryPrimOp(LTEQ(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(LTEQ(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(LTEQ(), f64, f64))
    eval(ApplyBinaryPrimOp(LTEQ(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(LTEQ(), f64, f64Zero))

    eval(ApplyBinaryPrimOp(EQ(), i32, i32))
    eval(ApplyBinaryPrimOp(EQ(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(EQ(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(EQ(), i64, i64))
    eval(ApplyBinaryPrimOp(EQ(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(EQ(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(EQ(), f32, f32))
    eval(ApplyBinaryPrimOp(EQ(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(EQ(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(EQ(), f64, f64))
    eval(ApplyBinaryPrimOp(EQ(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(EQ(), f64, f64Zero))

    eval(ApplyBinaryPrimOp(NEQ(), i32, i32))
    eval(ApplyBinaryPrimOp(NEQ(), i32Zero, i32))
    eval(ApplyBinaryPrimOp(NEQ(), i32, i32Zero))
    eval(ApplyBinaryPrimOp(NEQ(), i64, i64))
    eval(ApplyBinaryPrimOp(NEQ(), i64Zero, i64))
    eval(ApplyBinaryPrimOp(NEQ(), i64, i64Zero))
    eval(ApplyBinaryPrimOp(NEQ(), f32, f32))
    eval(ApplyBinaryPrimOp(NEQ(), f32Zero, f32))
    eval(ApplyBinaryPrimOp(NEQ(), f32, f32Zero))
    eval(ApplyBinaryPrimOp(NEQ(), f64, f64))
    eval(ApplyBinaryPrimOp(NEQ(), f64Zero, f64))
    eval(ApplyBinaryPrimOp(NEQ(), f64, f64Zero))
  }

  @Test def testCasts() {
    eval(Cast(i32, TInt32()))
    eval(Cast(i32, TInt64()))
    eval(Cast(i32, TFloat32()))
    eval(Cast(i32, TFloat64()))

    eval(Cast(i64, TInt32()))
    eval(Cast(i64, TInt64()))
    eval(Cast(i64, TFloat32()))
    eval(Cast(i64, TFloat64()))

    eval(Cast(f32, TInt32()))
    eval(Cast(f32, TInt64()))
    eval(Cast(f32, TFloat32()))
    eval(Cast(f32, TFloat64()))

    eval(Cast(f64, TInt32()))
    eval(Cast(f64, TInt64()))
    eval(Cast(f64, TFloat32()))
    eval(Cast(f64, TFloat64()))
  }

  @Test def testNA() {
    eval(NA(TInt32()))
    eval(NA(TStruct("a" -> TInt32(), "b" -> TString())))

    eval(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32))
    eval(ApplyBinaryPrimOp(EQ(), NA(TInt32()), i32))
  }

  @Test def testIsNA() {
    eval(IsNA(NA(TInt32())))
    eval(IsNA(NA(TStruct("a" -> TInt32(), "b" -> TString()))))
    eval(IsNA(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32)))
    eval(IsNA(ApplyBinaryPrimOp(EQ(), NA(TInt32()), i32)))
  }

  @Test def testIf() {
    eval(If(t, t, f))
    eval(If(t, f, f))
    eval(If(t, f, NA(TBoolean())))
    eval(If(t, Cast(i32, TFloat64()), f64))
  }

  @Test def testLet() {
    eval(Let("foo", i64, ApplyBinaryPrimOp(Add(), f64, Cast(Ref("foo", TInt64()), TFloat64()))))
  }

  @Test def testMakeArray() {
    eval(arr)
  }

  @Test def testArrayRef() {
    eval(ArrayRef(arr, I32(1)))
  }

  @Test def testArrayLen() {
    eval(ArrayLen(arr))
  }

  @Test def testArrayRange() {
    eval(ArrayRange(I32(0), I32(10), I32(2)))
    eval(ArrayRange(I32(0), I32(10), I32(1)))
    eval(ArrayRange(I32(0), I32(10), I32(3)))
  }

  @Test def testArrayMap() {
    eval(ArrayMap(arr, "foo", ApplyBinaryPrimOp(Multiply(), Ref("foo", TInt32()), Ref("foo", TInt32()))))
  }

  @Test def testArrayFilter() {
    eval(ArrayFilter(arr, "foo", ApplyBinaryPrimOp(LT(), Ref("foo", TInt32()), I32(2))))
    eval(ArrayFilter(arr, "foo", ApplyBinaryPrimOp(LT(), Ref("foo", TInt32()), NA(TInt32()))))
  }

  @Test def testArrayFlatMap() {
    eval(ArrayFlatMap(arr, "foo", ArrayRange(I32(-1), Ref("foo", TInt32()), I32(1))))
  }

  @Test def testArrayFold() {
    eval(ArrayFold(arr, I32(0), "sum", "element", ApplyBinaryPrimOp(Add(), Ref("sum", TInt32()), Ref("element", TInt32()))))
  }

  @Test def testMakeStruct() {
    eval(struct)
  }

  @Test def testInsertFields() {
    eval(InsertFields(struct, List("a" -> f64, "bar" -> i32)))
  }

  @Test def testGetField() {
    eval(GetField(struct, "a"))
  }

  @Test def testMakeTuple() {
    eval(tuple)
  }

  @Test def testGetTupleElement() {
    eval(GetTupleElement(tuple, 0))
    eval(GetTupleElement(tuple, 1))
    eval(GetTupleElement(tuple, 2))
  }

  @Test def testApplyMethods() {
    eval(Apply("log10", List(f64)))

    eval(ApplySpecial("||", List(t, f)))
    eval(ApplySpecial("||", List(t, t)))
    eval(ApplySpecial("||", List(f, f)))
    eval(ApplySpecial("&&", List(t, t)))
    eval(ApplySpecial("&&", List(t, f)))
    eval(ApplySpecial("&&", List(f, f)))
  }

  @Test def testAggregator() {
    val agg = (FastIndexedSeq(Row(5), Row(10), Row(15)),
      TStruct("a" -> TInt32()))
    val aggSig = AggSignature(Sum(), TInt64(), FastSeq(), None)
    assertEvalsTo(ApplyAggOp(
      If(ApplyBinaryPrimOp(LT(), Ref("a", TInt32()), I32(11)),
        SeqOp(Cast(Ref("a", TInt32()), TInt64()), I32(0), aggSig),
        Begin(FastIndexedSeq())),
      FastSeq(), None, aggSig),
      agg,
      15)
  }
}

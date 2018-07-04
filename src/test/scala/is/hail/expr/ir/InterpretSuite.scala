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
    assertEvalSame(t)
    assertEvalSame(f)
    assertEvalSame(ApplyUnaryPrimOp(Bang(), t))
    assertEvalSame(ApplyUnaryPrimOp(Bang(), f))

    assertEvalSame(ApplyUnaryPrimOp(Negate(), i32))
    assertEvalSame(ApplyUnaryPrimOp(Negate(), i64))
    assertEvalSame(ApplyUnaryPrimOp(Negate(), f32))
    assertEvalSame(ApplyUnaryPrimOp(Negate(), f64))
  }

  @Test def testApplyBinaryPrimOp() {
    assertEvalSame(i32)
    assertEvalSame(i64)
    assertEvalSame(f32)
    assertEvalSame(f64)

    assertEvalSame(ApplyBinaryPrimOp(Add(), i32, i32))
    assertEvalSame(ApplyBinaryPrimOp(Add(), i64, i64))
    assertEvalSame(ApplyBinaryPrimOp(Add(), f32, f32))
    assertEvalSame(ApplyBinaryPrimOp(Add(), f64, f64))

    assertEvalSame(ApplyBinaryPrimOp(Subtract(), i32, i32))
    assertEvalSame(ApplyBinaryPrimOp(Subtract(), i64, i64))
    assertEvalSame(ApplyBinaryPrimOp(Subtract(), f32, f32))
    assertEvalSame(ApplyBinaryPrimOp(Subtract(), f64, f64))

    assertEvalSame(ApplyBinaryPrimOp(FloatingPointDivide(), i32, i32))
    assertEvalSame(ApplyBinaryPrimOp(FloatingPointDivide(), i64, i64))
    assertEvalSame(ApplyBinaryPrimOp(FloatingPointDivide(), f32, f32))
    assertEvalSame(ApplyBinaryPrimOp(FloatingPointDivide(), f64, f64))

    assertEvalSame(ApplyBinaryPrimOp(RoundToNegInfDivide(), i32, i32))
    assertEvalSame(ApplyBinaryPrimOp(RoundToNegInfDivide(), i64, i64))
    assertEvalSame(ApplyBinaryPrimOp(RoundToNegInfDivide(), f32, f32))
    assertEvalSame(ApplyBinaryPrimOp(RoundToNegInfDivide(), f64, f64))

    assertEvalSame(ApplyBinaryPrimOp(Multiply(), i32, i32))
    assertEvalSame(ApplyBinaryPrimOp(Multiply(), i64, i64))
    assertEvalSame(ApplyBinaryPrimOp(Multiply(), f32, f32))
    assertEvalSame(ApplyBinaryPrimOp(Multiply(), f64, f64))
  }

  @Test def testApplyComparisonOp() {
    assertEvalSame(ApplyComparisonOp(EQ(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(EQ(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat64()), f64, f64))

    assertEvalSame(ApplyComparisonOp(GT(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(GT(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(GT(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(GT(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(GT(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(GT(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(GT(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(GT(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(GT(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(GT(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(GT(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(GT(TFloat64()), f64, f64Zero))

    assertEvalSame(ApplyComparisonOp(GTEQ(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(GTEQ(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(GTEQ(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(GTEQ(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(GTEQ(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(GTEQ(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(GTEQ(TFloat64()), f64, f64Zero))

    assertEvalSame(ApplyComparisonOp(LT(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(LT(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(LT(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(LT(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(LT(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(LT(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(LT(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(LT(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(LT(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(LT(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(LT(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(LT(TFloat64()), f64, f64Zero))

    assertEvalSame(ApplyComparisonOp(LTEQ(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(LTEQ(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(LTEQ(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(LTEQ(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(LTEQ(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(LTEQ(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(LTEQ(TFloat64()), f64, f64Zero))

    assertEvalSame(ApplyComparisonOp(EQ(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(EQ(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(EQ(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(EQ(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(EQ(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(EQ(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(EQ(TFloat64()), f64, f64Zero))

    assertEvalSame(ApplyComparisonOp(NEQ(TInt32()), i32, i32))
    assertEvalSame(ApplyComparisonOp(NEQ(TInt32()), i32Zero, i32))
    assertEvalSame(ApplyComparisonOp(NEQ(TInt32()), i32, i32Zero))
    assertEvalSame(ApplyComparisonOp(NEQ(TInt64()), i64, i64))
    assertEvalSame(ApplyComparisonOp(NEQ(TInt64()), i64Zero, i64))
    assertEvalSame(ApplyComparisonOp(NEQ(TInt64()), i64, i64Zero))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat32()), f32, f32))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat32()), f32Zero, f32))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat32()), f32, f32Zero))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat64()), f64, f64))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat64()), f64Zero, f64))
    assertEvalSame(ApplyComparisonOp(NEQ(TFloat64()), f64, f64Zero))
  }

  @Test def testCasts() {
    assertEvalSame(Cast(i32, TInt32()))
    assertEvalSame(Cast(i32, TInt64()))
    assertEvalSame(Cast(i32, TFloat32()))
    assertEvalSame(Cast(i32, TFloat64()))

    assertEvalSame(Cast(i64, TInt32()))
    assertEvalSame(Cast(i64, TInt64()))
    assertEvalSame(Cast(i64, TFloat32()))
    assertEvalSame(Cast(i64, TFloat64()))

    assertEvalSame(Cast(f32, TInt32()))
    assertEvalSame(Cast(f32, TInt64()))
    assertEvalSame(Cast(f32, TFloat32()))
    assertEvalSame(Cast(f32, TFloat64()))

    assertEvalSame(Cast(f64, TInt32()))
    assertEvalSame(Cast(f64, TInt64()))
    assertEvalSame(Cast(f64, TFloat32()))
    assertEvalSame(Cast(f64, TFloat64()))
  }

  @Test def testNA() {
    assertEvalSame(NA(TInt32()))
    assertEvalSame(NA(TStruct("a" -> TInt32(), "b" -> TString())))

    assertEvalSame(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32))
    assertEvalSame(ApplyComparisonOp(EQ(TInt32()), NA(TInt32()), i32))
  }

  @Test def testIsNA() {
    assertEvalSame(IsNA(NA(TInt32())))
    assertEvalSame(IsNA(NA(TStruct("a" -> TInt32(), "b" -> TString()))))
    assertEvalSame(IsNA(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32)))
    assertEvalSame(IsNA(ApplyComparisonOp(EQ(TInt32()), NA(TInt32()), i32)))
  }

  @Test def testIf() {
    assertEvalSame(If(t, t, f))
    assertEvalSame(If(t, f, f))
    assertEvalSame(If(t, f, NA(TBoolean())))
    assertEvalSame(If(t, Cast(i32, TFloat64()), f64))
  }

  @Test def testLet() {
    assertEvalSame(Let("foo", i64, ApplyBinaryPrimOp(Add(), f64, Cast(Ref("foo", TInt64()), TFloat64()))))
  }

  @Test def testMakeArray() {
    assertEvalSame(arr)
  }

  @Test def testArrayRef() {
    assertEvalSame(ArrayRef(arr, I32(1)))
  }

  @Test def testArrayLen() {
    assertEvalSame(ArrayLen(arr))
  }

  @Test def testArrayRange() {
    for {
      start <- -2 to 2
      stop <- -2 to 8
      step <- 1 to 3
    } {
      assertEvalSame(ArrayRange(I32(start), I32(stop), I32(step)))
      assertEvalSame(ArrayRange(I32(start), I32(stop), I32(-step)))
    }
    assertEvalSame(ArrayRange(I32(Int.MinValue), I32(Int.MaxValue), I32(Int.MaxValue / 5)))
  }

  @Test def testArrayMap() {
    assertEvalSame(ArrayMap(arr, "foo", ApplyBinaryPrimOp(Multiply(), Ref("foo", TInt32()), Ref("foo", TInt32()))))
  }

  @Test def testArrayFilter() {
    assertEvalSame(ArrayFilter(arr, "foo", ApplyComparisonOp(LT(TInt32()), Ref("foo", TInt32()), I32(2))))
    assertEvalSame(ArrayFilter(arr, "foo", ApplyComparisonOp(LT(TInt32()), Ref("foo", TInt32()), NA(TInt32()))))
  }

  @Test def testArrayFlatMap() {
    assertEvalSame(ArrayFlatMap(arr, "foo", ArrayRange(I32(-1), Ref("foo", TInt32()), I32(1))))
  }

  @Test def testArrayFold() {
    assertEvalSame(ArrayFold(arr, I32(0), "sum", "element", ApplyBinaryPrimOp(Add(), Ref("sum", TInt32()), Ref("element", TInt32()))))
  }

  @Test def testMakeStruct() {
    assertEvalSame(struct)
  }

  @Test def testInsertFields() {
    assertEvalSame(InsertFields(struct, List("a" -> f64, "bar" -> i32)))
  }

  @Test def testGetField() {
    assertEvalSame(GetField(struct, "a"))
  }

  @Test def testMakeTuple() {
    assertEvalSame(tuple)
  }

  @Test def testGetTupleElement() {
    assertEvalSame(GetTupleElement(tuple, 0))
    assertEvalSame(GetTupleElement(tuple, 1))
    assertEvalSame(GetTupleElement(tuple, 2))
  }

  @Test def testApplyMethods() {
    assertEvalSame(Apply("log10", List(f64)))

    assertEvalSame(ApplySpecial("||", List(t, f)))
    assertEvalSame(ApplySpecial("||", List(t, t)))
    assertEvalSame(ApplySpecial("||", List(f, f)))
    assertEvalSame(ApplySpecial("&&", List(t, t)))
    assertEvalSame(ApplySpecial("&&", List(t, f)))
    assertEvalSame(ApplySpecial("&&", List(f, f)))
  }

  @Test def testAggregator() {
    val agg = (FastIndexedSeq(Row(5), Row(10), Row(15)),
      TStruct("a" -> TInt32()))
    val aggSig = AggSignature(Sum(), FastSeq(), None, FastSeq(TInt64()))
    assertEvalsTo(ApplyAggOp(
      If(ApplyComparisonOp(LT(TInt32()), Ref("a", TInt32()), I32(11)),
        SeqOp(I32(0), FastSeq(Cast(Ref("a", TInt32()), TInt64())), aggSig),
        Begin(FastIndexedSeq())),
      FastSeq(), None, aggSig),
      agg,
      15)
  }
}

package is.hail.expr.ir

import is.hail.annotations.{Region, UnsafeRow}
import is.hail.expr.types._
import org.junit.Test

class InterpretSuite {

  private val env = Env.empty[(Any, Type)]
    .bind("a", (5, TInt32()))

  private val region = Region()

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

  private def check(ir: IR,
    env: Env[(Any, Type)] = env,
    args: IndexedSeq[(Any, Type)] = IndexedSeq(),
    agg: Option[(TAggregable, IndexedSeq[(Any, Env[Any])])] = None) {
    val interpreted = Interpret(ir, env, args, agg)
    val (rt, makeF) = Compile[Long](MakeTuple(List(ir)))
    val f = makeF()
    val compiled = new UnsafeRow(rt.asInstanceOf[TTuple], region, f(region)).get(0)
    assert(interpreted == compiled, s"\n  intr: $interpreted\n  comp: $compiled\n  IR: $ir")
  }

  @Test def testUnaryPrimOp() {
    check(t)
    check(f)
    check(ApplyUnaryPrimOp(Bang(), t))
    check(ApplyUnaryPrimOp(Bang(), f))

    check(ApplyUnaryPrimOp(Negate(), i32))
    check(ApplyUnaryPrimOp(Negate(), i64))
    check(ApplyUnaryPrimOp(Negate(), f32))
    check(ApplyUnaryPrimOp(Negate(), f64))
  }

  @Test def testApplyBinaryPrimOp() {
    check(i32)
    check(i64)
    check(f32)
    check(f64)

    check(ApplyBinaryPrimOp(Add(), i32, i32))
    check(ApplyBinaryPrimOp(Add(), i64, i64))
    check(ApplyBinaryPrimOp(Add(), f32, f32))
    check(ApplyBinaryPrimOp(Add(), f64, f64))

    check(ApplyBinaryPrimOp(Subtract(), i32, i32))
    check(ApplyBinaryPrimOp(Subtract(), i64, i64))
    check(ApplyBinaryPrimOp(Subtract(), f32, f32))
    check(ApplyBinaryPrimOp(Subtract(), f64, f64))

    check(ApplyBinaryPrimOp(FloatingPointDivide(), i32, i32))
    check(ApplyBinaryPrimOp(FloatingPointDivide(), i64, i64))
    check(ApplyBinaryPrimOp(FloatingPointDivide(), f32, f32))
    check(ApplyBinaryPrimOp(FloatingPointDivide(), f64, f64))

    check(ApplyBinaryPrimOp(RoundToNegInfDivide(), i32, i32))
    check(ApplyBinaryPrimOp(RoundToNegInfDivide(), i64, i64))
    check(ApplyBinaryPrimOp(RoundToNegInfDivide(), f32, f32))
    check(ApplyBinaryPrimOp(RoundToNegInfDivide(), f64, f64))

    check(ApplyBinaryPrimOp(Multiply(), i32, i32))
    check(ApplyBinaryPrimOp(Multiply(), i64, i64))
    check(ApplyBinaryPrimOp(Multiply(), f32, f32))
    check(ApplyBinaryPrimOp(Multiply(), f64, f64))

    check(ApplyBinaryPrimOp(EQ(), i32, i32))
    check(ApplyBinaryPrimOp(EQ(), i64, i64))
    check(ApplyBinaryPrimOp(EQ(), f32, f32))
    check(ApplyBinaryPrimOp(EQ(), f64, f64))

    check(ApplyBinaryPrimOp(GT(), i32, i32))
    check(ApplyBinaryPrimOp(GT(), i32Zero, i32))
    check(ApplyBinaryPrimOp(GT(), i32, i32Zero))
    check(ApplyBinaryPrimOp(GT(), i64, i64))
    check(ApplyBinaryPrimOp(GT(), i64Zero, i64))
    check(ApplyBinaryPrimOp(GT(), i64, i64Zero))
    check(ApplyBinaryPrimOp(GT(), f32, f32))
    check(ApplyBinaryPrimOp(GT(), f32Zero, f32))
    check(ApplyBinaryPrimOp(GT(), f32, f32Zero))
    check(ApplyBinaryPrimOp(GT(), f64, f64))
    check(ApplyBinaryPrimOp(GT(), f64Zero, f64))
    check(ApplyBinaryPrimOp(GT(), f64, f64Zero))

    check(ApplyBinaryPrimOp(GTEQ(), i32, i32))
    check(ApplyBinaryPrimOp(GTEQ(), i32Zero, i32))
    check(ApplyBinaryPrimOp(GTEQ(), i32, i32Zero))
    check(ApplyBinaryPrimOp(GTEQ(), i64, i64))
    check(ApplyBinaryPrimOp(GTEQ(), i64Zero, i64))
    check(ApplyBinaryPrimOp(GTEQ(), i64, i64Zero))
    check(ApplyBinaryPrimOp(GTEQ(), f32, f32))
    check(ApplyBinaryPrimOp(GTEQ(), f32Zero, f32))
    check(ApplyBinaryPrimOp(GTEQ(), f32, f32Zero))
    check(ApplyBinaryPrimOp(GTEQ(), f64, f64))
    check(ApplyBinaryPrimOp(GTEQ(), f64Zero, f64))
    check(ApplyBinaryPrimOp(GTEQ(), f64, f64Zero))

    check(ApplyBinaryPrimOp(LT(), i32, i32))
    check(ApplyBinaryPrimOp(LT(), i32Zero, i32))
    check(ApplyBinaryPrimOp(LT(), i32, i32Zero))
    check(ApplyBinaryPrimOp(LT(), i64, i64))
    check(ApplyBinaryPrimOp(LT(), i64Zero, i64))
    check(ApplyBinaryPrimOp(LT(), i64, i64Zero))
    check(ApplyBinaryPrimOp(LT(), f32, f32))
    check(ApplyBinaryPrimOp(LT(), f32Zero, f32))
    check(ApplyBinaryPrimOp(LT(), f32, f32Zero))
    check(ApplyBinaryPrimOp(LT(), f64, f64))
    check(ApplyBinaryPrimOp(LT(), f64Zero, f64))
    check(ApplyBinaryPrimOp(LT(), f64, f64Zero))

    check(ApplyBinaryPrimOp(LTEQ(), i32, i32))
    check(ApplyBinaryPrimOp(LTEQ(), i32Zero, i32))
    check(ApplyBinaryPrimOp(LTEQ(), i32, i32Zero))
    check(ApplyBinaryPrimOp(LTEQ(), i64, i64))
    check(ApplyBinaryPrimOp(LTEQ(), i64Zero, i64))
    check(ApplyBinaryPrimOp(LTEQ(), i64, i64Zero))
    check(ApplyBinaryPrimOp(LTEQ(), f32, f32))
    check(ApplyBinaryPrimOp(LTEQ(), f32Zero, f32))
    check(ApplyBinaryPrimOp(LTEQ(), f32, f32Zero))
    check(ApplyBinaryPrimOp(LTEQ(), f64, f64))
    check(ApplyBinaryPrimOp(LTEQ(), f64Zero, f64))
    check(ApplyBinaryPrimOp(LTEQ(), f64, f64Zero))

    check(ApplyBinaryPrimOp(EQ(), i32, i32))
    check(ApplyBinaryPrimOp(EQ(), i32Zero, i32))
    check(ApplyBinaryPrimOp(EQ(), i32, i32Zero))
    check(ApplyBinaryPrimOp(EQ(), i64, i64))
    check(ApplyBinaryPrimOp(EQ(), i64Zero, i64))
    check(ApplyBinaryPrimOp(EQ(), i64, i64Zero))
    check(ApplyBinaryPrimOp(EQ(), f32, f32))
    check(ApplyBinaryPrimOp(EQ(), f32Zero, f32))
    check(ApplyBinaryPrimOp(EQ(), f32, f32Zero))
    check(ApplyBinaryPrimOp(EQ(), f64, f64))
    check(ApplyBinaryPrimOp(EQ(), f64Zero, f64))
    check(ApplyBinaryPrimOp(EQ(), f64, f64Zero))

    check(ApplyBinaryPrimOp(NEQ(), i32, i32))
    check(ApplyBinaryPrimOp(NEQ(), i32Zero, i32))
    check(ApplyBinaryPrimOp(NEQ(), i32, i32Zero))
    check(ApplyBinaryPrimOp(NEQ(), i64, i64))
    check(ApplyBinaryPrimOp(NEQ(), i64Zero, i64))
    check(ApplyBinaryPrimOp(NEQ(), i64, i64Zero))
    check(ApplyBinaryPrimOp(NEQ(), f32, f32))
    check(ApplyBinaryPrimOp(NEQ(), f32Zero, f32))
    check(ApplyBinaryPrimOp(NEQ(), f32, f32Zero))
    check(ApplyBinaryPrimOp(NEQ(), f64, f64))
    check(ApplyBinaryPrimOp(NEQ(), f64Zero, f64))
    check(ApplyBinaryPrimOp(NEQ(), f64, f64Zero))
  }

  @Test def testCasts() {
    check(Cast(i32, TInt32()))
    check(Cast(i32, TInt64()))
    check(Cast(i32, TFloat32()))
    check(Cast(i32, TFloat64()))

    check(Cast(i64, TInt32()))
    check(Cast(i64, TInt64()))
    check(Cast(i64, TFloat32()))
    check(Cast(i64, TFloat64()))

    check(Cast(f32, TInt32()))
    check(Cast(f32, TInt64()))
    check(Cast(f32, TFloat32()))
    check(Cast(f32, TFloat64()))

    check(Cast(f64, TInt32()))
    check(Cast(f64, TInt64()))
    check(Cast(f64, TFloat32()))
    check(Cast(f64, TFloat64()))
  }

  @Test def testNA() {
    check(NA(TInt32()))
    check(NA(TStruct("a" -> TInt32(), "b" -> TString())))

    check(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32))
    check(ApplyBinaryPrimOp(EQ(), NA(TInt32()), i32))
  }

  @Test def testIsNA() {
    check(IsNA(NA(TInt32())))
    check(IsNA(NA(TStruct("a" -> TInt32(), "b" -> TString()))))
    check(IsNA(ApplyBinaryPrimOp(Add(), NA(TInt32()), i32)))
    check(IsNA(ApplyBinaryPrimOp(EQ(), NA(TInt32()), i32)))
  }

  @Test def testIf() {
    check(If(t, t, f))
    check(If(t, f, f))
    check(If(t, f, NA(TBoolean())))
    check(If(t, Cast(i32, TFloat64()), f64))
  }

  @Test def testLet() {
    check(Let("foo", i64, ApplyBinaryPrimOp(Add(), f64, Cast(Ref("foo", TInt64()), TFloat64()))))
  }

  @Test def testMakeArray() {
    check(arr)
  }

  @Test def testArrayRef() {
    check(ArrayRef(arr, I32(1)))
  }

  @Test def testArrayLen() {
    check(ArrayLen(arr))
  }

  @Test def testArrayRange() {
    check(ArrayRange(I32(0), I32(10), I32(2)))
    check(ArrayRange(I32(0), I32(10), I32(1)))
    check(ArrayRange(I32(0), I32(10), I32(3)))
  }

  @Test def testArrayMap() {
    check(ArrayMap(arr, "foo", ApplyBinaryPrimOp(Multiply(), Ref("foo", TInt32()), Ref("foo", TInt32()))))
  }

  @Test def testArrayFilter() {
    check(ArrayFilter(arr, "foo", ApplyBinaryPrimOp(LT(), Ref("foo", TInt32()), I32(2))))
    check(ArrayFilter(arr, "foo", ApplyBinaryPrimOp(LT(), Ref("foo", TInt32()), NA(TInt32()))))
  }

  @Test def testArrayFlatMap() {
    check(ArrayFlatMap(arr, "foo", ArrayRange(I32(-1), Ref("foo", TInt32()), I32(1))))
  }

  @Test def testArrayFold() {
    check(ArrayFold(arr, I32(0), "sum", "element", ApplyBinaryPrimOp(Add(), Ref("sum", TInt32()), Ref("element", TInt32()))))
  }

  @Test def testMakeStruct() {
    check(struct)
  }

  @Test def testInsertFields() {
    check(InsertFields(struct, List("a" -> f64, "bar" -> i32)))
  }

  @Test def testGetField() {
    check(GetField(struct, "a"))
  }

  @Test def testMakeTuple() {
    check(tuple)
  }

  @Test def testGetTupleElement() {
    check(GetTupleElement(tuple, 0))
    check(GetTupleElement(tuple, 1))
    check(GetTupleElement(tuple, 2))
  }

  @Test def testApplyMethods() {
    check(Apply("log10", List(f64)))

    check(ApplySpecial("||", List(t, f)))
    check(ApplySpecial("||", List(t, t)))
    check(ApplySpecial("||", List(f, f)))
    check(ApplySpecial("&&", List(t, t)))
    check(ApplySpecial("&&", List(t, f)))
    check(ApplySpecial("&&", List(f, f)))
  }

  @Test def testAggregator() {
    val aggT = TAggregable(TInt32(), Map("a" -> (0 -> TInt32())))
    val agg = aggT -> IndexedSeq(
      5 -> Env.empty[Any].bind("a" -> 10),
      10 -> Env.empty[Any].bind("a" -> 20),
      15 -> Env.empty[Any].bind("a" -> 30)
    )

    val result = Interpret(ApplyAggOp(AggFilter(AggIn(aggT), "x", ApplyBinaryPrimOp(LT(), Ref("a", TInt32()), I32(21))),
      Sum(), List()), env, IndexedSeq(), Some(agg))
    assert(result == 15)
  }
}

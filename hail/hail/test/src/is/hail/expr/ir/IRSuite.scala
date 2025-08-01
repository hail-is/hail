package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.annotations.{BroadcastRow, ExtendedOrdering, Region, RegionPool, SafeNDArray}
import is.hail.backend.ExecuteContext
import is.hail.expr.Nat
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.agg._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.defs.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.ir.functions._
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.io.bgen.MatrixBGENReader
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.rvd.{PartitionBoundOrdering, RVD, RVDPartitioner}
import is.hail.types.{tcoerce, VirtualTypeWithReq}
import is.hail.types.physical._
import is.hail.types.physical.stypes._
import is.hail.types.physical.stypes.primitives.SInt32
import is.hail.types.virtual._
import is.hail.types.virtual.TIterable.elementType
import is.hail.utils.{FastSeq, _}
import is.hail.variant.{Call2, Locus}

import scala.collection.mutable
import scala.language.implicitConversions

import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.scalatest
import org.scalatest.Inspectors.forAll
import org.scalatest.enablers.InspectorAsserting.assertingNatureOfAssertion
import org.testng.annotations.{DataProvider, Test}

class IRSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.nonLowering

  @Test def testRandDifferentLengthUIDStrings(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.lowering
    val staticUID: Long = 112233
    var rng: IR = RNGStateLiteral()
    rng = RNGSplit(rng, I64(12345))
    val expected1 = Threefry.pmac(ctx.rngNonce, staticUID, Array(12345L))
    assertEvalsTo(ApplySeeded("rand_int64", IndexedSeq(), rng, staticUID, TInt64), expected1(0))

    rng = RNGSplit(rng, I64(0))
    val expected2 = Threefry.pmac(ctx.rngNonce, staticUID, Array(12345L, 0L))
    assertEvalsTo(ApplySeeded("rand_int64", IndexedSeq(), rng, staticUID, TInt64), expected2(0))

    rng = RNGSplit(rng, I64(0))
    rng = RNGSplit(rng, I64(0))
    val expected3 = Threefry.pmac(ctx.rngNonce, staticUID, Array(12345L, 0L, 0L, 0L))
    assertEvalsTo(ApplySeeded("rand_int64", IndexedSeq(), rng, staticUID, TInt64), expected3(0))
    assert(expected1 != expected2)
    assert(expected2 != expected3)
    assert(expected1 != expected3)
  }

  @Test def testI32(): scalatest.Assertion =
    assertEvalsTo(I32(5), 5)

  @Test def testI64(): scalatest.Assertion =
    assertEvalsTo(I64(5), 5L)

  @Test def testF32(): scalatest.Assertion =
    assertEvalsTo(F32(3.14f), 3.14f)

  @Test def testF64(): scalatest.Assertion =
    assertEvalsTo(F64(3.14), 3.14)

  @Test def testStr(): scalatest.Assertion =
    assertEvalsTo(Str("Hail"), "Hail")

  @Test def testTrue(): scalatest.Assertion =
    assertEvalsTo(True(), true)

  @Test def testFalse(): scalatest.Assertion =
    assertEvalsTo(False(), false)
  // FIXME Void() doesn't work because we can't handle a void type in a tuple

  @Test def testCast(): scalatest.Assertion = {
    assertAllEvalTo(
      (Cast(I32(5), TInt32), 5),
      (Cast(I32(5), TInt64), 5L),
      (Cast(I32(5), TFloat32), 5.0f),
      (Cast(I32(5), TFloat64), 5.0),
      (Cast(I64(5), TInt32), 5),
      (Cast(I64(0xf29fb5c9af12107dL), TInt32), 0xaf12107d), // truncate
      (Cast(I64(5), TInt64), 5L),
      (Cast(I64(5), TFloat32), 5.0f),
      (Cast(I64(5), TFloat64), 5.0),
      (Cast(F32(3.14f), TInt32), 3),
      (Cast(F32(3.99f), TInt32), 3), // truncate
      (Cast(F32(3.14f), TInt64), 3L),
      (Cast(F32(3.14f), TFloat32), 3.14f),
      (Cast(F32(3.14f), TFloat64), 3.14),
      (Cast(F64(3.14), TInt32), 3),
      (Cast(F64(3.99), TInt32), 3), // truncate
      (Cast(F64(3.14), TInt64), 3L),
      (Cast(F64(3.14), TFloat32), 3.14f),
      (Cast(F64(3.14), TFloat64), 3.14),
    )
  }

  @Test def testCastRename(): scalatest.Assertion = {
    assertEvalsTo(CastRename(MakeStruct(FastSeq(("x", I32(1)))), TStruct("foo" -> TInt32)), Row(1))
    assertEvalsTo(
      CastRename(
        MakeArray(FastSeq(MakeStruct(FastSeq(("x", I32(1))))), TArray(TStruct("x" -> TInt32))),
        TArray(TStruct("foo" -> TInt32)),
      ),
      FastSeq(Row(1)),
    )
  }

  @Test def testNA(): scalatest.Assertion =
    assertEvalsTo(NA(TInt32), null)

  @Test def testCoalesce(): scalatest.Assertion = {
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32))), FastSeq((null, TInt32)), null)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32))), FastSeq((1, TInt32)), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), In(0, TInt32))), FastSeq((null, TInt32)), null)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), In(0, TInt32))), FastSeq((1, TInt32)), 1)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32), NA(TInt32))), FastSeq((1, TInt32)), 1)
    assertEvalsTo(
      Coalesce(FastSeq(NA(TInt32), I32(1), I32(1), NA(TInt32), I32(1), NA(TInt32), I32(1))),
      1,
    )
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), I32(1), Die("foo", TInt32))), 1)
  }

  @Test def testCoalesceWithDifferentRequiredeness(): scalatest.Assertion = {
    val t1 = In(0, TArray(TInt32))
    val t2 = NA(TArray(TInt32))
    val value = FastSeq(1, 2, 3, 4)

    assertEvalsTo(Coalesce(FastSeq(t1, t2)), FastSeq((value, TArray(TInt32))), value)
  }

  val i32na = NA(TInt32)
  val i64na = NA(TInt64)
  val f32na = NA(TFloat32)
  val f64na = NA(TFloat64)
  val bna = NA(TBoolean)

  @Test def testApplyUnaryPrimOpNegate(): scalatest.Assertion = {
    assertAllEvalTo(
      (ApplyUnaryPrimOp(Negate, I32(5)), -5),
      (ApplyUnaryPrimOp(Negate, i32na), null),
      (ApplyUnaryPrimOp(Negate, I64(5)), -5L),
      (ApplyUnaryPrimOp(Negate, i64na), null),
      (ApplyUnaryPrimOp(Negate, F32(5)), -5f),
      (ApplyUnaryPrimOp(Negate, f32na), null),
      (ApplyUnaryPrimOp(Negate, F64(5)), -5d),
      (ApplyUnaryPrimOp(Negate, f64na), null),
    )
  }

  @Test def testApplyUnaryPrimOpBang(): scalatest.Assertion = {
    assertEvalsTo(ApplyUnaryPrimOp(Bang, False()), true)
    assertEvalsTo(ApplyUnaryPrimOp(Bang, True()), false)
    assertEvalsTo(ApplyUnaryPrimOp(Bang, bna), null)
  }

  @Test def testApplyUnaryPrimOpBitFlip(): scalatest.Assertion = {
    assertAllEvalTo(
      (ApplyUnaryPrimOp(BitNot, I32(0xdeadbeef)), ~0xdeadbeef),
      (ApplyUnaryPrimOp(BitNot, I32(-0xdeadbeef)), ~(-0xdeadbeef)),
      (ApplyUnaryPrimOp(BitNot, i32na), null),
      (ApplyUnaryPrimOp(BitNot, I64(0xdeadbeef12345678L)), ~0xdeadbeef12345678L),
      (ApplyUnaryPrimOp(BitNot, I64(-0xdeadbeef12345678L)), ~(-0xdeadbeef12345678L)),
      (ApplyUnaryPrimOp(BitNot, i64na), null),
    )
  }

  @Test def testApplyUnaryPrimOpBitCount(): scalatest.Assertion = {
    assertAllEvalTo(
      (ApplyUnaryPrimOp(BitCount, I32(0xdeadbeef)), Integer.bitCount(0xdeadbeef)),
      (ApplyUnaryPrimOp(BitCount, I32(-0xdeadbeef)), Integer.bitCount(-0xdeadbeef)),
      (ApplyUnaryPrimOp(BitCount, i32na), null),
      (
        ApplyUnaryPrimOp(BitCount, I64(0xdeadbeef12345678L)),
        java.lang.Long.bitCount(0xdeadbeef12345678L),
      ),
      (
        ApplyUnaryPrimOp(BitCount, I64(-0xdeadbeef12345678L)),
        java.lang.Long.bitCount(-0xdeadbeef12345678L),
      ),
      (ApplyUnaryPrimOp(BitCount, i64na), null),
    )
  }

  @Test def testApplyBinaryPrimOpAdd(): scalatest.Assertion = {
    def assertSumsTo(t: Type, x: Any, y: Any, sum: Any): scalatest.Assertion =
      assertEvalsTo(ApplyBinaryPrimOp(Add(), In(0, t), In(1, t)), FastSeq(x -> t, y -> t), sum)
    assertSumsTo(TInt32, 5, 3, 8)
    assertSumsTo(TInt32, 5, null, null)
    assertSumsTo(TInt32, null, 3, null)
    assertSumsTo(TInt32, null, null, null)

    assertSumsTo(TInt64, 5L, 3L, 8L)
    assertSumsTo(TInt64, 5L, null, null)
    assertSumsTo(TInt64, null, 3L, null)
    assertSumsTo(TInt64, null, null, null)

    assertSumsTo(TFloat32, 5.0f, 3.0f, 8.0f)
    assertSumsTo(TFloat32, 5.0f, null, null)
    assertSumsTo(TFloat32, null, 3.0f, null)
    assertSumsTo(TFloat32, null, null, null)

    assertSumsTo(TFloat64, 5.0, 3.0, 8.0)
    assertSumsTo(TFloat64, 5.0, null, null)
    assertSumsTo(TFloat64, null, 3.0, null)
    assertSumsTo(TFloat64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpSubtract(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(Subtract(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 3)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 3L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)

    assertExpected(TFloat32, 5f, 2f, 3f)
    assertExpected(TFloat32, 5f, null, null)
    assertExpected(TFloat32, null, 2f, null)
    assertExpected(TFloat32, null, null, null)

    assertExpected(TFloat64, 5d, 2d, 3d)
    assertExpected(TFloat64, 5d, null, null)
    assertExpected(TFloat64, null, 2d, null)
    assertExpected(TFloat64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpMultiply(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(Multiply(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 10)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 10L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)

    assertExpected(TFloat32, 5f, 2f, 10f)
    assertExpected(TFloat32, 5f, null, null)
    assertExpected(TFloat32, null, 2f, null)
    assertExpected(TFloat32, null, null, null)

    assertExpected(TFloat64, 5d, 2d, 10d)
    assertExpected(TFloat64, 5d, null, null)
    assertExpected(TFloat64, null, 2d, null)
    assertExpected(TFloat64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpFloatingPointDivide(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(FloatingPointDivide(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 2.5)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 2.5)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)

    assertExpected(TFloat32, 5f, 2f, 2.5f)
    assertExpected(TFloat32, 5f, null, null)
    assertExpected(TFloat32, null, 2f, null)
    assertExpected(TFloat32, null, null, null)

    assertExpected(TFloat64, 5d, 2d, 2.5d)
    assertExpected(TFloat64, 5d, null, null)
    assertExpected(TFloat64, null, 2d, null)
    assertExpected(TFloat64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpRoundToNegInfDivide(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(RoundToNegInfDivide(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 2)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 2L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)

    assertExpected(TFloat32, 5f, 2f, 2f)
    assertExpected(TFloat32, 5f, null, null)
    assertExpected(TFloat32, null, 2f, null)
    assertExpected(TFloat32, null, null, null)

    assertExpected(TFloat64, 5d, 2d, 2d)
    assertExpected(TFloat64, 5d, null, null)
    assertExpected(TFloat64, null, 2d, null)
    assertExpected(TFloat64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitAnd(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(BitAnd(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 5 & 2)
    assertExpected(TInt32, -5, 2, -5 & 2)
    assertExpected(TInt32, 5, -2, 5 & -2)
    assertExpected(TInt32, -5, -2, -5 & -2)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 5L & 2L)
    assertExpected(TInt64, -5L, 2L, -5L & 2L)
    assertExpected(TInt64, 5L, -2L, 5L & -2L)
    assertExpected(TInt64, -5L, -2L, -5L & -2L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitOr(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(BitOr(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 5 | 2)
    assertExpected(TInt32, -5, 2, -5 | 2)
    assertExpected(TInt32, 5, -2, 5 | -2)
    assertExpected(TInt32, -5, -2, -5 | -2)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 5L | 2L)
    assertExpected(TInt64, -5L, 2L, -5L | 2L)
    assertExpected(TInt64, 5L, -2L, 5L | -2L)
    assertExpected(TInt64, -5L, -2L, -5L | -2L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitXOr(): scalatest.Assertion = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(BitXOr(), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertExpected(TInt32, 5, 2, 5 ^ 2)
    assertExpected(TInt32, -5, 2, -5 ^ 2)
    assertExpected(TInt32, 5, -2, 5 ^ -2)
    assertExpected(TInt32, -5, -2, -5 ^ -2)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 5L ^ 2L)
    assertExpected(TInt64, -5L, 2L, -5L ^ 2L)
    assertExpected(TInt64, 5L, -2L, 5L ^ -2L)
    assertExpected(TInt64, -5L, -2L, -5L ^ -2L)
    assertExpected(TInt64, 5L, null, null)
    assertExpected(TInt64, null, 2L, null)
    assertExpected(TInt64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpLeftShift(): scalatest.Assertion = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(LeftShift(), In(0, t), In(1, TInt32)),
        FastSeq(x -> t, y -> TInt32),
        expected,
      )

    assertShiftsTo(TInt32, 5, 2, 5 << 2)
    assertShiftsTo(TInt32, -5, 2, -5 << 2)
    assertShiftsTo(TInt32, 5, null, null)
    assertShiftsTo(TInt32, null, 2, null)
    assertShiftsTo(TInt32, null, null, null)

    assertShiftsTo(TInt64, 5L, 2, 5L << 2)
    assertShiftsTo(TInt64, -5L, 2, -5L << 2)
    assertShiftsTo(TInt64, 5L, null, null)
    assertShiftsTo(TInt64, null, 2, null)
    assertShiftsTo(TInt64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpRightShift(): scalatest.Assertion = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(RightShift(), In(0, t), In(1, TInt32)),
        FastSeq(x -> t, y -> TInt32),
        expected,
      )

    assertShiftsTo(TInt32, 0xff5, 2, 0xff5 >> 2)
    assertShiftsTo(TInt32, -5, 2, -5 >> 2)
    assertShiftsTo(TInt32, 5, null, null)
    assertShiftsTo(TInt32, null, 2, null)
    assertShiftsTo(TInt32, null, null, null)

    assertShiftsTo(TInt64, 0xffff5L, 2, 0xffff5L >> 2)
    assertShiftsTo(TInt64, -5L, 2, -5L >> 2)
    assertShiftsTo(TInt64, 5L, null, null)
    assertShiftsTo(TInt64, null, 2, null)
    assertShiftsTo(TInt64, null, null, null)
  }

  @Test def testApplyBinaryPrimOpLogicalRightShift(): scalatest.Assertion = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any): scalatest.Assertion =
      assertEvalsTo(
        ApplyBinaryPrimOp(LogicalRightShift(), In(0, t), In(1, TInt32)),
        FastSeq(x -> t, y -> TInt32),
        expected,
      )

    assertShiftsTo(TInt32, 0xff5, 2, 0xff5 >>> 2)
    assertShiftsTo(TInt32, -5, 2, -5 >>> 2)
    assertShiftsTo(TInt32, 5, null, null)
    assertShiftsTo(TInt32, null, 2, null)
    assertShiftsTo(TInt32, null, null, null)

    assertShiftsTo(TInt64, 0xffff5L, 2, 0xffff5L >>> 2)
    assertShiftsTo(TInt64, -5L, 2, -5L >>> 2)
    assertShiftsTo(TInt64, 5L, null, null)
    assertShiftsTo(TInt64, null, 2, null)
    assertShiftsTo(TInt64, null, null, null)
  }

  @Test def testApplyComparisonOpGT(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(ApplyComparisonOp(GT(t), In(0, t), In(1, t)), FastSeq(x -> t, y -> t), expected)

    assertComparesTo(TInt32, 1, 1, false)
    assertComparesTo(TInt32, 0, 1, false)
    assertComparesTo(TInt32, 1, 0, true)

    assertComparesTo(TInt64, 1L, 1L, false)
    assertComparesTo(TInt64, 0L, 1L, false)
    assertComparesTo(TInt64, 1L, 0L, true)

    assertComparesTo(TFloat32, 1.0f, 1.0f, false)
    assertComparesTo(TFloat32, 0.0f, 1.0f, false)
    assertComparesTo(TFloat32, 1.0f, 0.0f, true)

    assertComparesTo(TFloat64, 1.0, 1.0, false)
    assertComparesTo(TFloat64, 0.0, 1.0, false)
    assertComparesTo(TFloat64, 1.0, 0.0, true)

  }

  @Test def testApplyComparisonOpGTEQ(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(
        ApplyComparisonOp(GTEQ(t), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertComparesTo(TInt32, 1, 1, true)
    assertComparesTo(TInt32, 0, 1, false)
    assertComparesTo(TInt32, 1, 0, true)

    assertComparesTo(TInt64, 1L, 1L, true)
    assertComparesTo(TInt64, 0L, 1L, false)
    assertComparesTo(TInt64, 1L, 0L, true)

    assertComparesTo(TFloat32, 1.0f, 1.0f, true)
    assertComparesTo(TFloat32, 0.0f, 1.0f, false)
    assertComparesTo(TFloat32, 1.0f, 0.0f, true)

    assertComparesTo(TFloat64, 1.0, 1.0, true)
    assertComparesTo(TFloat64, 0.0, 1.0, false)
    assertComparesTo(TFloat64, 1.0, 0.0, true)
  }

  @Test def testApplyComparisonOpLT(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(ApplyComparisonOp(LT(t), In(0, t), In(1, t)), FastSeq(x -> t, y -> t), expected)

    assertComparesTo(TInt32, 1, 1, false)
    assertComparesTo(TInt32, 0, 1, true)
    assertComparesTo(TInt32, 1, 0, false)

    assertComparesTo(TInt64, 1L, 1L, false)
    assertComparesTo(TInt64, 0L, 1L, true)
    assertComparesTo(TInt64, 1L, 0L, false)

    assertComparesTo(TFloat32, 1.0f, 1.0f, false)
    assertComparesTo(TFloat32, 0.0f, 1.0f, true)
    assertComparesTo(TFloat32, 1.0f, 0.0f, false)

    assertComparesTo(TFloat64, 1.0, 1.0, false)
    assertComparesTo(TFloat64, 0.0, 1.0, true)
    assertComparesTo(TFloat64, 1.0, 0.0, false)

  }

  @Test def testApplyComparisonOpLTEQ(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(
        ApplyComparisonOp(LTEQ(t), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertComparesTo(TInt32, 1, 1, true)
    assertComparesTo(TInt32, 0, 1, true)
    assertComparesTo(TInt32, 1, 0, false)

    assertComparesTo(TInt64, 1L, 1L, true)
    assertComparesTo(TInt64, 0L, 1L, true)
    assertComparesTo(TInt64, 1L, 0L, false)

    assertComparesTo(TFloat32, 1.0f, 1.0f, true)
    assertComparesTo(TFloat32, 0.0f, 1.0f, true)
    assertComparesTo(TFloat32, 1.0f, 0.0f, false)

    assertComparesTo(TFloat64, 1.0, 1.0, true)
    assertComparesTo(TFloat64, 0.0, 1.0, true)
    assertComparesTo(TFloat64, 1.0, 0.0, false)

  }

  @Test def testApplyComparisonOpEQ(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(ApplyComparisonOp(EQ(t), In(0, t), In(1, t)), FastSeq(x -> t, y -> t), expected)

    assertComparesTo(TInt32, 1, 1, expected = true)
    assertComparesTo(TInt32, 0, 1, expected = false)
    assertComparesTo(TInt32, 1, 0, expected = false)

    assertComparesTo(TInt64, 1L, 1L, expected = true)
    assertComparesTo(TInt64, 0L, 1L, expected = false)
    assertComparesTo(TInt64, 1L, 0L, expected = false)

    assertComparesTo(TFloat32, 1.0f, 1.0f, expected = true)
    assertComparesTo(TFloat32, 0.0f, 1.0f, expected = false)
    assertComparesTo(TFloat32, 1.0f, 0.0f, expected = false)

    assertComparesTo(TFloat64, 1.0, 1.0, expected = true)
    assertComparesTo(TFloat64, 0.0, 1.0, expected = false)
    assertComparesTo(TFloat64, 1.0, 0.0, expected = false)
  }

  @Test def testApplyComparisonOpNE(): scalatest.Assertion = {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean): scalatest.Assertion =
      assertEvalsTo(
        ApplyComparisonOp(NEQ(t), In(0, t), In(1, t)),
        FastSeq(x -> t, y -> t),
        expected,
      )

    assertComparesTo(TInt32, 1, 1, expected = false)
    assertComparesTo(TInt32, 0, 1, expected = true)
    assertComparesTo(TInt32, 1, 0, expected = true)

    assertComparesTo(TInt64, 1L, 1L, expected = false)
    assertComparesTo(TInt64, 0L, 1L, expected = true)
    assertComparesTo(TInt64, 1L, 0L, expected = true)

    assertComparesTo(TFloat32, 1.0f, 1.0f, expected = false)
    assertComparesTo(TFloat32, 0.0f, 1.0f, expected = true)
    assertComparesTo(TFloat32, 1.0f, 0.0f, expected = true)

    assertComparesTo(TFloat64, 1.0, 1.0, expected = false)
    assertComparesTo(TFloat64, 0.0, 1.0, expected = true)
    assertComparesTo(TFloat64, 1.0, 0.0, expected = true)
  }

  @Test def testDieCodeBUilder(): scalatest.Assertion =
    assertFatal(Die("msg1", TInt32) + Die("msg2", TInt32), "msg1")

  @Test def testIf(): scalatest.Assertion = {
    assertEvalsTo(If(True(), I32(5), I32(7)), 5)
    assertEvalsTo(If(False(), I32(5), I32(7)), 7)
    assertEvalsTo(If(NA(TBoolean), I32(5), I32(7)), null)
    assertEvalsTo(If(True(), NA(TInt32), I32(7)), null)
  }

  @DataProvider(name = "SwitchEval")
  def switchEvalRules: Array[Array[Any]] =
    Array(
      Array(I32(-1), I32(Int.MinValue), FastSeq(0, Int.MaxValue).map(I32), Int.MinValue),
      Array(I32(0), I32(Int.MinValue), FastSeq(0, Int.MaxValue).map(I32), 0),
      Array(I32(1), I32(Int.MinValue), FastSeq(0, Int.MaxValue).map(I32), Int.MaxValue),
      Array(I32(2), I32(Int.MinValue), FastSeq(0, Int.MaxValue).map(I32), Int.MinValue),
      Array(NA(TInt32), I32(Int.MinValue), FastSeq(0, Int.MaxValue).map(I32), null),
      Array(I32(-1), NA(TInt32), FastSeq(0, Int.MaxValue).map(I32), null),
      Array(I32(0), NA(TInt32), FastSeq(NA(TInt32), I32(0)), null),
    )

  @Test(dataProvider = "SwitchEval")
  def testSwitch(x: IR, default: IR, cases: IndexedSeq[IR], result: Any): scalatest.Assertion =
    assertEvalsTo(Switch(x, default, cases), result)

  @Test def testLet(): scalatest.Assertion = {
    assertEvalsTo(bindIR(I32(5))(x => x), 5)
    assertEvalsTo(bindIR(NA(TInt32))(x => x), null)
    assertEvalsTo(bindIR(I32(5))(_ => NA(TInt32)), null)
    assertEvalsTo(
      ToArray(mapIR(bindIR(I32(5))(v => rangeIR(v)))(x => x + I32(2))),
      FastSeq(2, 3, 4, 5, 6),
    )
    assertEvalsTo(
      ToArray(mapIR(
        bindIR(I32(2))(q => mapIR(bindIR(q + I32(3))(v => rangeIR(v)))(x => x + q))
      )(y => y + I32(3))),
      FastSeq(5, 6, 7, 8, 9),
    )

    // test let binding streams
    assertEvalsTo(
      bindIR(MakeStream(IndexedSeq(I32(0), I32(5)), TStream(TInt32)))(ToArray),
      FastSeq(0, 5),
    )
    assertEvalsTo(
      bindIR(NA(TStream(TInt32)))(ToArray),
      null,
    )
    assertEvalsTo(
      ToArray(bindIR(MakeStream(IndexedSeq(I32(0), I32(5)), TStream(TInt32))) { s =>
        StreamTake(s, I32(1))
      }),
      FastSeq(0),
    )
  }

  @Test def testMakeArray(): scalatest.Assertion = {
    assertEvalsTo(
      MakeArray(FastSeq(I32(5), NA(TInt32), I32(-3)), TArray(TInt32)),
      FastSeq(5, null, -3),
    )
    assertEvalsTo(MakeArray(FastSeq(), TArray(TInt32)), FastSeq())
  }

  @Test def testGetNestedElementPTypesI32(): scalatest.Assertion = {
    var types = IndexedSeq(PInt32(true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PInt32(true))

    types = IndexedSeq(PInt32(false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt32(false))

    types = IndexedSeq(PInt32(false), PInt32(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt32(false))

    types = IndexedSeq(PInt32(true), PInt32(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt32(true))
  }

  @Test def testGetNestedElementPTypesI64(): scalatest.Assertion = {
    var types = IndexedSeq(PInt64(true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PInt64(true))

    types = IndexedSeq(PInt64(false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt64(false))

    types = IndexedSeq(PInt64(false), PInt64(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt64(false))

    types = IndexedSeq(PInt64(true), PInt64(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PInt64(true))
  }

  @Test def testGetNestedElementPFloat32(): scalatest.Assertion = {
    var types = IndexedSeq(PFloat32(true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(true))

    types = IndexedSeq(PFloat32(false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(false))

    types = IndexedSeq(PFloat32(false), PFloat32(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(false))

    types = IndexedSeq(PFloat32(true), PFloat32(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(true))
  }

  @Test def testGetNestedElementPFloat64(): scalatest.Assertion = {
    var types = IndexedSeq(PFloat64(true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(true))

    types = IndexedSeq(PFloat64(false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(false))

    types = IndexedSeq(PFloat64(false), PFloat64(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(false))

    types = IndexedSeq(PFloat64(true), PFloat64(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(true))
  }

  @Test def testGetNestedElementPCanonicalString(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalString(true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(true))

    types = IndexedSeq(PCanonicalString(false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(false))

    types = IndexedSeq(PCanonicalString(false), PCanonicalString(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(false))

    types = IndexedSeq(PCanonicalString(true), PCanonicalString(true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(true))
  }

  @Test def testGetNestedPCanonicalArray(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalArray(PInt32(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), true))

    types = IndexedSeq(PCanonicalArray(PInt32(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), false))

    types = IndexedSeq(PCanonicalArray(PInt32(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), true))

    types = IndexedSeq(PCanonicalArray(PInt32(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalArray(PInt32(true), true),
      PCanonicalArray(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), true))

    types = IndexedSeq(
      PCanonicalArray(PInt32(false), true),
      PCanonicalArray(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), true))

    types = IndexedSeq(
      PCanonicalArray(PInt32(false), true),
      PCanonicalArray(PInt32(true), false),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(true), true), true))

    types = IndexedSeq(
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), true), true))

    types = IndexedSeq(
      PCanonicalArray(PCanonicalArray(PInt32(true), false), true),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), false), true))

    types = IndexedSeq(
      PCanonicalArray(PCanonicalArray(PInt32(true), false), false),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), false), false))
  }

  @Test def testGetNestedElementPCanonicalDict(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalDict(PInt32(true), PCanonicalString(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), true))

    types = IndexedSeq(PCanonicalDict(PInt32(false), PCanonicalString(true), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(true), true))

    types = IndexedSeq(PCanonicalDict(PInt32(true), PCanonicalString(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(false), true))

    types = IndexedSeq(PCanonicalDict(PInt32(true), PCanonicalString(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), false))

    types = IndexedSeq(PCanonicalDict(PInt32(false), PCanonicalString(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), false))

    types = IndexedSeq(
      PCanonicalDict(PInt32(true), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), true))

    types = IndexedSeq(
      PCanonicalDict(PInt32(true), PCanonicalString(true), false),
      PCanonicalDict(PInt32(true), PCanonicalString(true), false),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), false))

    types = IndexedSeq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(true), true))

    types = IndexedSeq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(false), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), true))

    types = IndexedSeq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), false),
      PCanonicalDict(PInt32(true), PCanonicalString(false), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), false))

    types = IndexedSeq(
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(true), PCanonicalString(true), true),
        true,
      ),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(
      PInt32(true),
      PCanonicalDict(PInt32(true), PCanonicalString(true), true),
      true,
    ))

    types = IndexedSeq(
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(false), PCanonicalString(true), true),
        true,
      ),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(
      PInt32(true),
      PCanonicalDict(PInt32(false), PCanonicalString(true), true),
      true,
    ))

    types = IndexedSeq(
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(false), PCanonicalString(true), true),
        true,
      ),
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(true), PCanonicalString(false), true),
        true,
      ),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(
      PInt32(true),
      PCanonicalDict(PInt32(false), PCanonicalString(false), true),
      true,
    ))

    types = IndexedSeq(
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(false), PCanonicalString(true), true),
        true,
      ),
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(true), PCanonicalString(false), true),
        true,
      ),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(
      PInt32(true),
      PCanonicalDict(PInt32(false), PCanonicalString(false), true),
      true,
    ))

    types = IndexedSeq(
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(false), PCanonicalString(true), false),
        true,
      ),
      PCanonicalDict(
        PInt32(true),
        PCanonicalDict(PInt32(true), PCanonicalString(false), true),
        true,
      ),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(
      PInt32(true),
      PCanonicalDict(PInt32(false), PCanonicalString(false), false),
      true,
    ))
  }

  @Test def testGetNestedElementPCanonicalStruct(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = IndexedSeq(PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = IndexedSeq(PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(true)))

    types = IndexedSeq(PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(false)))

    types = IndexedSeq(PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = IndexedSeq(
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = IndexedSeq(
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = IndexedSeq(
      PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = IndexedSeq(
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)),
        "b" -> PInt32(true),
      )
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(
      true,
      "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)),
      "b" -> PInt32(true),
    ))

    types = IndexedSeq(
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(true)),
        "b" -> PInt32(true),
      )
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(
      true,
      "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(true)),
      "b" -> PInt32(true),
    ))

    types = IndexedSeq(
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(false)),
        "b" -> PInt32(true),
      ),
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)),
        "b" -> PInt32(true),
      ),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(
      true,
      "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(false)),
      "b" -> PInt32(true),
    ))

    types = IndexedSeq(
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(false, "c" -> PInt32(false), "d" -> PInt32(false)),
        "b" -> PInt32(true),
      ),
      PCanonicalStruct(
        true,
        "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)),
        "b" -> PInt32(true),
      ),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(
      true,
      "a" -> PCanonicalStruct(false, "c" -> PInt32(false), "d" -> PInt32(false)),
      "b" -> PInt32(true),
    ))
  }

  @Test def testGetNestedElementPCanonicalTuple(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalTuple(true, PInt32(true)))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(true)))

    types = IndexedSeq(PCanonicalTuple(false, PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(true)))

    types = IndexedSeq(PCanonicalTuple(true, PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(false)))

    types = IndexedSeq(PCanonicalTuple(false, PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(false)))

    types = IndexedSeq(
      PCanonicalTuple(true, PInt32(true)),
      PCanonicalTuple(true, PInt32(true)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(true)))

    types = IndexedSeq(
      PCanonicalTuple(true, PInt32(true)),
      PCanonicalTuple(false, PInt32(true)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(true)))

    types = IndexedSeq(
      PCanonicalTuple(true, PInt32(false)),
      PCanonicalTuple(false, PInt32(true)),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(false)))

    types = IndexedSeq(
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(true))),
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false))),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false))))

    types = IndexedSeq(
      PCanonicalTuple(true, PCanonicalTuple(false, PInt32(true))),
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false))),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PCanonicalTuple(false, PInt32(false))))
  }

  @Test def testGetNestedElementPCanonicalSet(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalSet(PInt32(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), true))

    types = IndexedSeq(PCanonicalSet(PInt32(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), false))

    types = IndexedSeq(PCanonicalSet(PInt32(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), true))

    types = IndexedSeq(PCanonicalSet(PInt32(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalSet(PInt32(true), true),
      PCanonicalSet(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), true))

    types = IndexedSeq(
      PCanonicalSet(PInt32(false), true),
      PCanonicalSet(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), true))

    types = IndexedSeq(
      PCanonicalSet(PInt32(false), true),
      PCanonicalSet(PInt32(true), false),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true),
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(true), true), true))

    types = IndexedSeq(
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true),
      PCanonicalSet(PCanonicalSet(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(false), true), true))

    types = IndexedSeq(
      PCanonicalSet(PCanonicalSet(PInt32(true), false), true),
      PCanonicalSet(PCanonicalSet(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(false), false), true))
  }

  @Test def testGetNestedElementPCanonicalInterval(): scalatest.Assertion = {
    var types = IndexedSeq(PCanonicalInterval(PInt32(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), true))

    types = IndexedSeq(PCanonicalInterval(PInt32(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), false))

    types = IndexedSeq(PCanonicalInterval(PInt32(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), true))

    types = IndexedSeq(PCanonicalInterval(PInt32(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalInterval(PInt32(true), true),
      PCanonicalInterval(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), true))

    types = IndexedSeq(
      PCanonicalInterval(PInt32(false), true),
      PCanonicalInterval(PInt32(true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), true))

    types = IndexedSeq(
      PCanonicalInterval(PInt32(true), true),
      PCanonicalInterval(PInt32(true), false),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), false))

    types = IndexedSeq(
      PCanonicalInterval(PInt32(false), true),
      PCanonicalInterval(PInt32(true), false),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), false))

    types = IndexedSeq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true))

    types = IndexedSeq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true))

    types = IndexedSeq(
      PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true))

    types = IndexedSeq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true),
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(false), false), true))
  }

  @Test def testMakeStruct(): scalatest.Assertion = {
    assertEvalsTo(MakeStruct(FastSeq()), Row())
    assertEvalsTo(MakeStruct(FastSeq("a" -> NA(TInt32), "b" -> 4, "c" -> 0.5)), Row(null, 4, 0.5))
    // making sure wide structs get emitted without failure
    assertEvalsTo(GetField(MakeStruct((0 until 20000).map(i => s"foo$i" -> I32(1))), "foo1"), 1)
  }

  @Test def testMakeArrayWithDifferentRequiredness(): scalatest.Assertion = {
    val pt1 = PCanonicalArray(PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalArray(PInt32())))
    val pt2 = PCanonicalArray(PCanonicalStruct(
      true,
      "a" -> PInt32(true),
      "b" -> PCanonicalArray(PInt32(), true),
    ))

    val value = Row(2, FastSeq(1))
    assertEvalsTo(
      MakeArray(
        In(0, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pt1.elementType))),
        In(1, SingleCodeEmitParamType(true, PTypeReferenceSingleCodeType(pt2.elementType))),
      ),
      FastSeq((null, pt1.virtualType.elementType), (value, pt2.virtualType.elementType)),
      FastSeq(null, value),
    )
  }

  @Test def testMakeTuple(): scalatest.Assertion = {
    assertEvalsTo(MakeTuple.ordered(FastSeq()), Row())
    assertEvalsTo(MakeTuple.ordered(FastSeq(NA(TInt32), 4, 0.5)), Row(null, 4, 0.5))
    // making sure wide structs get emitted without failure
    assertEvalsTo(GetTupleElement(MakeTuple.ordered((0 until 20000).map(I32)), 1), 1)
  }

  @Test def testGetTupleElement(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = MakeTuple.ordered(FastSeq(I32(5), Str("abc"), NA(TInt32)))
    val na = NA(TTuple(TInt32, TString))

    assertEvalsTo(GetTupleElement(t, 0), 5)
    assertEvalsTo(GetTupleElement(t, 1), "abc")
    assertEvalsTo(GetTupleElement(t, 2), null)
    assertEvalsTo(GetTupleElement(na, 0), null)
  }

  @Test def testLetBoundPrunedTuple(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.unoptimizedCompileOnly
    val t2 = MakeTuple(FastSeq((2, I32(5))))

    val letBoundTuple = bindIR(t2)(tupleRef => GetTupleElement(tupleRef, 2))

    assertEvalsTo(letBoundTuple, 5)
  }

  @Test def testArrayRef(): scalatest.Assertion = {
    assertEvalsTo(
      ArrayRef(MakeArray(FastSeq(I32(5), NA(TInt32)), TArray(TInt32)), I32(0), ErrorIDs.NO_ERROR),
      5,
    )
    assertEvalsTo(
      ArrayRef(MakeArray(FastSeq(I32(5), NA(TInt32)), TArray(TInt32)), I32(1), ErrorIDs.NO_ERROR),
      null,
    )
    assertEvalsTo(
      ArrayRef(
        MakeArray(FastSeq(I32(5), NA(TInt32)), TArray(TInt32)),
        NA(TInt32),
        ErrorIDs.NO_ERROR,
      ),
      null,
    )

    assertFatal(
      ArrayRef(MakeArray(FastSeq(I32(5)), TArray(TInt32)), I32(2)),
      "array index out of bounds",
    )
  }

  @Test def testArrayLen(): scalatest.Assertion = {
    assertEvalsTo(ArrayLen(NA(TArray(TInt32))), null)
    assertEvalsTo(ArrayLen(MakeArray(FastSeq(), TArray(TInt32))), 0)
    assertEvalsTo(ArrayLen(MakeArray(FastSeq(I32(5), NA(TInt32)), TArray(TInt32))), 2)
  }

  @Test def testArraySort(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ArraySort(ToStream(NA(TArray(TInt32)))), null)

    val a = MakeArray(FastSeq(I32(-7), I32(2), NA(TInt32), I32(2)), TArray(TInt32))
    assertEvalsTo(ArraySort(ToStream(a)), FastSeq(-7, 2, 2, null))
    assertEvalsTo(ArraySort(ToStream(a), False()), FastSeq(2, 2, -7, null))
  }

  @Test def testStreamZip(): scalatest.Assertion = {
    val range12 = StreamRange(0, 12, 1)
    val range6 = StreamRange(0, 12, 2)
    val range8 = StreamRange(0, 24, 3)
    val empty = StreamRange(0, 0, 1)
    val lit6 = ToStream(Literal(TArray(TFloat64), FastSeq(0d, -1d, 2.5d, -3d, 4d, null)))
    val range6dup = StreamRange(0, 6, 1)

    def zip(behavior: ArrayZipBehavior, irs: IR*): IR =
      zipIR(irs.toFastSeq, behavior)(MakeTuple.ordered)

    def zipToTuple(behavior: ArrayZipBehavior, irs: IR*): IR =
      ToArray(zip(behavior, irs: _*))

    for (b <- Array(ArrayZipBehavior.TakeMinLength, ArrayZipBehavior.ExtendNA)) {
      assertEvalSame(zipToTuple(b, range12), FastSeq())
      assertEvalSame(zipToTuple(b, range6, range8), FastSeq())
      assertEvalSame(zipToTuple(b, range6, range8), FastSeq())
      assertEvalSame(zipToTuple(b, range6, range8, lit6), FastSeq())
      assertEvalSame(zipToTuple(b, range12, lit6), FastSeq())
      assertEvalSame(zipToTuple(b, range12, lit6, empty), FastSeq())
      assertEvalSame(zipToTuple(b, empty, lit6), FastSeq())
      assertEvalSame(zipToTuple(b, empty), FastSeq())
    }

    for (b <- Array(ArrayZipBehavior.AssumeSameLength, ArrayZipBehavior.AssertSameLength)) {
      assertEvalSame(zipToTuple(b, range6, lit6), FastSeq())
      assertEvalSame(zipToTuple(b, range6, lit6, range6dup), FastSeq())
      assertEvalSame(zipToTuple(b, range12), FastSeq())
      assertEvalSame(zipToTuple(b, empty), FastSeq())
    }

    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.TakeMinLength, range6, range8)), 6)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.ExtendNA, range6, range8)), 8)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.AssertSameLength, range8, range8)), 8)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.AssumeSameLength, range8, range8)), 8)

    // https://github.com/hail-is/hail/issues/8359
    assertThrows[HailException](
      zipToTuple(ArrayZipBehavior.AssertSameLength, range6, range8): IR,
      "zip: length mismatch": String,
    )
    assertThrows[HailException](
      zipToTuple(ArrayZipBehavior.AssertSameLength, range12, lit6): IR,
      "zip: length mismatch": String,
    )
  }

  @Test def testToSet(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToSet(ToStream(NA(TArray(TInt32)))), null)
    assertEvalsTo(ToSet(NA(TStream(TInt32))), null)

    val a = MakeArray(FastSeq(I32(-7), I32(2), NA(TInt32), I32(2)), TArray(TInt32))
    assertEvalsTo(ToSet(ToStream(a)), Set(-7, 2, null))
  }

  @Test def testToArrayFromSet(): scalatest.Assertion = {
    val t = TSet(TInt32)
    assertEvalsTo(CastToArray(NA(t)), null)
    assertEvalsTo(CastToArray(In(0, t)), FastSeq((Set(-7, 2, null), t)), FastSeq(-7, 2, null))
  }

  @Test def testToDict(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToDict(ToStream(NA(TArray(TTuple(FastSeq(TInt32, TString): _*))))), null)

    val a = MakeArray(
      FastSeq(
        MakeTuple.ordered(FastSeq(I32(5), Str("a"))),
        MakeTuple.ordered(FastSeq(I32(5), Str("a"))), // duplicate key-value pair
        MakeTuple.ordered(FastSeq(NA(TInt32), Str("b"))),
        MakeTuple.ordered(FastSeq(I32(3), NA(TString))),
        NA(TTuple(FastSeq(TInt32, TString): _*)), // missing value
      ),
      TArray(TTuple(FastSeq(TInt32, TString): _*)),
    )

    assertEvalsTo(ToDict(ToStream(a)), Map(5 -> "a", (null, "b"), 3 -> null))
  }

  @Test def testToArrayFromDict(): scalatest.Assertion = {
    val t = TDict(TInt32, TString)
    assertEvalsTo(CastToArray(NA(t)), null)

    val d: Map[Any, Any] = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(
      CastToArray(In(0, t)),
      // wtf you can't do null -> ...
      FastSeq((d, t)),
      FastSeq(Row(1, "a"), Row(2, null), Row(null, "c")),
    )
  }

  @Test def testToArrayFromArray(): scalatest.Assertion = {
    val t = TArray(TInt32)
    assertEvalsTo(NA(t), null)
    assertEvalsTo(In(0, t), FastSeq((FastSeq(-7, 2, null, 2), t)), FastSeq(-7, 2, null, 2))
  }

  @Test def testSetContains(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TSet(TInt32)
    assertEvalsTo(invoke("contains", TBoolean, NA(t), I32(2)), null)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(TInt32)),
      FastSeq((Set(-7, 2, null), t)),
      true,
    )
    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), I32(2)),
      FastSeq((Set(-7, 2, null), t)),
      true,
    )
    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), I32(0)),
      FastSeq((Set(-7, 2, null), t)),
      false,
    )
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(7)), FastSeq((Set(-7, 2), t)), false)
  }

  @Test def testDictContains(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TDict(TInt32, TString)
    assertEvalsTo(invoke("contains", TBoolean, NA(t), I32(2)), null)

    val d: Map[Any, Any] = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), NA(TInt32)), FastSeq((d, t)), true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(2)), FastSeq((d, t)), true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(0)), FastSeq((d, t)), false)
    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), I32(3)),
      FastSeq((Map(1 -> "a", 2 -> null), t)),
      false,
    )
  }

  @Test def testLowerBoundOnOrderedCollectionArray(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TArray(TInt32))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val awoutna = MakeArray(FastSeq(I32(0), I32(2), I32(4)), TArray(TInt32))
    val awna = MakeArray(FastSeq(I32(0), I32(2), I32(4), NA(TInt32)), TArray(TInt32))
    val awdups =
      MakeArray(FastSeq(I32(0), I32(0), I32(2), I32(4), I32(4), NA(TInt32)), TArray(TInt32))
    assertAllEvalTo(
      (LowerBoundOnOrderedCollection(awoutna, I32(-1), onKey = false), 0),
      (LowerBoundOnOrderedCollection(awoutna, I32(0), onKey = false), 0),
      (LowerBoundOnOrderedCollection(awoutna, I32(1), onKey = false), 1),
      (LowerBoundOnOrderedCollection(awoutna, I32(2), onKey = false), 1),
      (LowerBoundOnOrderedCollection(awoutna, I32(3), onKey = false), 2),
      (LowerBoundOnOrderedCollection(awoutna, I32(4), onKey = false), 2),
      (LowerBoundOnOrderedCollection(awoutna, I32(5), onKey = false), 3),
      (LowerBoundOnOrderedCollection(awoutna, NA(TInt32), onKey = false), 3),
      (LowerBoundOnOrderedCollection(awna, NA(TInt32), onKey = false), 3),
      (LowerBoundOnOrderedCollection(awna, I32(5), onKey = false), 3),
      (LowerBoundOnOrderedCollection(awdups, I32(0), onKey = false), 0),
      (LowerBoundOnOrderedCollection(awdups, I32(4), onKey = false), 3),
    )
  }

  @Test def testLowerBoundOnOrderedCollectionSet(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TSet(TInt32))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val swoutna = ToSet(MakeStream(FastSeq(I32(0), I32(2), I32(4), I32(4)), TStream(TInt32)))
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(-1), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(0), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(1), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(2), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(3), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(4), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(5), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, NA(TInt32), onKey = false), 3)

    val swna =
      ToSet(MakeStream(FastSeq(I32(0), I32(2), I32(2), I32(4), NA(TInt32)), TStream(TInt32)))
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, NA(TInt32), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, I32(5), onKey = false), 3)
  }

  @Test def testLowerBoundOnOrderedCollectionDict(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TDict(TInt32, TString))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = true), null)

    val dwna = IRDict((1, 3), (3, null), (null, 5))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(2), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(3), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(5), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, NA(TInt32), onKey = true), 2)

    val dwoutna = IRDict((1, 3), (3, null))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(4), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, NA(TInt32), onKey = true), 2)
  }

  @Test def testStreamLen(): scalatest.Assertion = {
    val a = StreamLen(MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32)))
    assertEvalsTo(a, 3)

    val missing = StreamLen(NA(TStream(TInt64)))
    assertEvalsTo(missing, null)

    val range1 = StreamLen(StreamRange(1, 11, 1))
    assertEvalsTo(range1, 10)
    val range2 = StreamLen(StreamRange(20, 40, 2))
    assertEvalsTo(range2, 10)
    val range3 = StreamLen(StreamRange(-10, 5, 1))
    assertEvalsTo(range3, 15)
    val mappedRange = StreamLen(mapIR(StreamRange(2, 7, 1))(ref => maxIR(ref, 3)))
    assertEvalsTo(mappedRange, 5)

    val streamOfStreams = mapIR(rangeIR(5))(elementRef => rangeIR(elementRef))
    assertEvalsTo(StreamLen(flatMapIR(streamOfStreams)(x => x)), 4 + 3 + 2 + 1)

    val filteredRange = StreamLen(filterIR(rangeIR(12))(_ < I32(5)))
    assertEvalsTo(filteredRange, 5)

    val lenOfLet = StreamLen(bindIR(I32(5))(ref =>
      StreamGrouped(
        mapIR(rangeIR(20))(range_element =>
          InsertFields(MakeStruct(IndexedSeq(("num", range_element + ref))), IndexedSeq(("y", 12)))
        ),
        3,
      )
    ))
    assertEvalsTo(lenOfLet, 7)
  }

  @Test def testStreamLenUnconsumedInnerStream(): scalatest.Assertion =
    assertEvalsTo(
      StreamLen(
        mapIR(StreamGrouped(filterIR(rangeIR(10))(x => x.cne(I32(0))), 3))(group => ToArray(group))
      ),
      3,
    )

  @Test def testStreamTake(): scalatest.Assertion = {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamTake(naa, I32(2))), null)
    assertEvalsTo(ToArray(StreamTake(a, NA(TInt32))), null)
    assertEvalsTo(ToArray(StreamTake(a, I32(0))), FastSeq())
    assertEvalsTo(ToArray(StreamTake(a, I32(2))), FastSeq(3, null))
    assertEvalsTo(ToArray(StreamTake(a, I32(5))), FastSeq(3, null, 7))
    assertFatal(ToArray(StreamTake(a, I32(-1))), "stream take: negative num")
    assertEvalsTo(StreamLen(StreamTake(a, 2)), 2)
  }

  @Test def testStreamDrop(): scalatest.Assertion = {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamDrop(naa, I32(2))), null)
    assertEvalsTo(ToArray(StreamDrop(a, NA(TInt32))), null)
    assertEvalsTo(ToArray(StreamDrop(a, I32(0))), FastSeq(3, null, 7))
    assertEvalsTo(ToArray(StreamDrop(a, I32(2))), FastSeq(7))
    assertEvalsTo(ToArray(StreamDrop(a, I32(5))), FastSeq())
    assertFatal(ToArray(StreamDrop(a, I32(-1))), "stream drop: negative num")

    assertEvalsTo(StreamLen(StreamDrop(a, 1)), 2)
  }

  def toNestedArray(stream: IR): IR = ToArray(mapIR(stream)(ToArray))

  @Test def testStreamGrouped(): scalatest.Assertion = {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(toNestedArray(StreamGrouped(naa, I32(2))), null)
    assertEvalsTo(toNestedArray(StreamGrouped(a, NA(TInt32))), null)
    assertEvalsTo(
      toNestedArray(StreamGrouped(MakeStream(IndexedSeq(), TStream(TInt32)), I32(2))),
      FastSeq(),
    )
    assertEvalsTo(
      toNestedArray(StreamGrouped(a, I32(1))),
      FastSeq(FastSeq(3), FastSeq(null), FastSeq(7)),
    )
    assertEvalsTo(toNestedArray(StreamGrouped(a, I32(2))), FastSeq(FastSeq(3, null), FastSeq(7)))
    assertEvalsTo(toNestedArray(StreamGrouped(a, I32(5))), FastSeq(FastSeq(3, null, 7)))
    assertFatal(toNestedArray(StreamGrouped(a, I32(0))), "stream grouped: non-positive size")

    val r = rangeIR(10)

    // test when inner streams are unused
    assertEvalsTo(streamForceCount(StreamGrouped(rangeIR(10), 2)), 5)

    assertEvalsTo(StreamLen(StreamGrouped(r, 2)), 5)

    def takeFromEach(stream: IR, take: IR, fromEach: IR): IR =
      mapIR(StreamGrouped(stream, fromEach))(StreamTake(_, take))

    assertEvalsTo(
      toNestedArray(takeFromEach(r, I32(1), I32(3))),
      FastSeq(FastSeq(0), FastSeq(3), FastSeq(6), FastSeq(9)),
    )
    assertEvalsTo(
      toNestedArray(takeFromEach(r, I32(2), I32(3))),
      FastSeq(FastSeq(0, 1), FastSeq(3, 4), FastSeq(6, 7), FastSeq(9)),
    )
    assertEvalsTo(toNestedArray(takeFromEach(r, I32(0), I32(5))), FastSeq(FastSeq(), FastSeq()))
  }

  @Test def testStreamGroupByKey(): scalatest.Assertion = {
    val structType = TStruct("a" -> TInt32, "b" -> TInt32)
    val naa = NA(TStream(structType))
    val a = MakeStream(
      IndexedSeq(
        MakeStruct(IndexedSeq("a" -> I32(3), "b" -> I32(1))),
        MakeStruct(IndexedSeq("a" -> I32(3), "b" -> I32(3))),
        MakeStruct(IndexedSeq("a" -> NA(TInt32), "b" -> I32(-1))),
        MakeStruct(IndexedSeq("a" -> NA(TInt32), "b" -> I32(-2))),
        MakeStruct(IndexedSeq("a" -> I32(1), "b" -> I32(2))),
        MakeStruct(IndexedSeq("a" -> I32(1), "b" -> I32(4))),
        MakeStruct(IndexedSeq("a" -> I32(1), "b" -> I32(6))),
        MakeStruct(IndexedSeq("a" -> I32(4), "b" -> NA(TInt32))),
      ),
      TStream(structType),
    )

    def group(a: IR): IR = StreamGroupByKey(a, FastSeq("a"), false)
    assertEvalsTo(toNestedArray(group(naa)), null)
    assertEvalsTo(
      toNestedArray(group(a)),
      FastSeq(
        FastSeq(Row(3, 1), Row(3, 3)),
        FastSeq(Row(null, -1)),
        FastSeq(Row(null, -2)),
        FastSeq(Row(1, 2), Row(1, 4), Row(1, 6)),
        FastSeq(Row(4, null)),
      ),
    )
    assertEvalsTo(toNestedArray(group(MakeStream(IndexedSeq(), TStream(structType)))), FastSeq())

    // test when inner streams are unused
    assertEvalsTo(streamForceCount(group(a)), 5)

    def takeFromEach(stream: IR, take: IR): IR =
      mapIR(group(stream))(StreamTake(_, take))

    assertEvalsTo(
      toNestedArray(takeFromEach(a, I32(1))),
      FastSeq(
        FastSeq(Row(3, 1)),
        FastSeq(Row(null, -1)),
        FastSeq(Row(null, -2)),
        FastSeq(Row(1, 2)),
        FastSeq(Row(4, null)),
      ),
    )
    assertEvalsTo(
      toNestedArray(takeFromEach(a, I32(2))),
      FastSeq(
        FastSeq(Row(3, 1), Row(3, 3)),
        FastSeq(Row(null, -1)),
        FastSeq(Row(null, -2)),
        FastSeq(Row(1, 2), Row(1, 4)),
        FastSeq(Row(4, null)),
      ),
    )
  }

  @Test def testStreamMap(): scalatest.Assertion = {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(mapIR(naa)(_ => I32(5))), null)
    assertEvalsTo(ToArray(mapIR(a)(_ + I32(1))), FastSeq(4, null, 8))
  }

  @Test def testStreamFilter(): scalatest.Assertion = {
    val nsa = NA(TStream(TInt32))
    val a = MakeStream(IndexedSeq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(filterIR(nsa)(_ => True())), null)

    assertEvalsTo(ToArray(filterIR(a)(_ => NA(TBoolean))), FastSeq())
    assertEvalsTo(ToArray(filterIR(a)(_ => False())), FastSeq())
    assertEvalsTo(ToArray(filterIR(a)(_ => True())), FastSeq(3, null, 7))

    assertEvalsTo(ToArray(filterIR(a)(IsNA)), FastSeq(null))
    assertEvalsTo(ToArray(filterIR(a)(!IsNA(_))), FastSeq(3, 7))

    assertEvalsTo(ToArray(filterIR(a)(_ < I32(6))), FastSeq(3))
  }

  @Test def testArrayFlatMap(): scalatest.Assertion = {
    val ta = TArray(TInt32)
    val ts = TStream(TInt32)
    val tsa = TStream(ta)
    val nsa = NA(tsa)
    val naas = MakeStream(FastSeq(NA(ta), NA(ta)), tsa)
    val a = MakeStream(
      FastSeq(
        MakeArray(FastSeq(I32(7), NA(TInt32)), ta),
        NA(ta),
        MakeArray(FastSeq(I32(2)), ta),
      ),
      tsa,
    )

    assertEvalsTo(ToArray(flatMapIR(nsa)(_ => MakeStream(FastSeq(I32(5)), ts))), null)

    assertEvalsTo(ToArray(flatMapIR(naas)(ToStream(_))), FastSeq())

    assertEvalsTo(ToArray(flatMapIR(a)(ToStream(_))), FastSeq(7, null, 2))

    assertEvalsTo(
      ToArray(flatMapIR(rangeIR(3))(i => ToStream(ArrayRef(ToArray(a), i)))),
      FastSeq(7, null, 2),
    )

    val st = MakeStream(FastSeq(I32(1), I32(5), I32(2), NA(TInt32)), TStream(TInt32))
    val expected = FastSeq(-1, 0, -1, 0, 1, 2, 3, 4, -1, 0, 1)
    assertEvalsTo(ToArray(flatMapIR(st)(foo => rangeIR(-1, foo))), expected)
  }

  @Test def testStreamFold(): scalatest.Assertion = {
    assertEvalsTo(foldIR(StreamRange(1, 2, 1), NA(TBoolean))((accum, elt) => IsNA(accum)), true)
    assertEvalsTo(foldIR(IRStream(1, 2, 3), 0)((accum, elt) => accum + elt), 6)
    assertEvalsTo(
      foldIR(IRStream(1, 2, 3), NA(TInt32))((accum, elt) => accum + elt),
      null,
    )
    assertEvalsTo(
      foldIR(IRStream(1, null, 3), NA(TInt32))((accum, elt) => accum + elt),
      null,
    )
    assertEvalsTo(foldIR(IRStream(1, null, 3), 0)((accum, elt) => accum + elt), null)
    assertEvalsTo(
      foldIR(IRStream(1, null, 3), NA(TInt32))((accum, elt) => I32(5) + I32(5)),
      10,
    )
  }

  @Test def testArrayFold2(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    val af = fold2IR(
      ToStream(In(0, TArray(TInt32))),
      I32(0),
      NA(TInt32),
    )(
      { case (elt, Seq(x, _)) =>
        elt + x
      },
      { case (elt, Seq(_, y)) =>
        Coalesce(FastSeq(y, elt))
      },
    ) { case Seq(x, y) =>
      MakeStruct(FastSeq(("x", x), ("y", y)))
    }

    assertEvalsTo(af, FastSeq((FastSeq(1, 2, 3), TArray(TInt32))), Row(6, 1))
  }

  @Test def testArrayScan(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(
      ToArray(streamScanIR(StreamRange(1, 4, 1), NA(TBoolean))((accum, elt) => IsNA(accum))),
      FastSeq(null, true, false, false),
    )
    assertEvalsTo(
      ToArray(streamScanIR(IRStream(1, 2, 3), 0)((accum, elt) => accum + elt)),
      FastSeq(0, 1, 3, 6),
    )
    assertEvalsTo(
      ToArray(streamScanIR(IRStream(1, 2, 3), NA(TInt32))((accum, elt) => accum + elt)),
      FastSeq(null, null, null, null),
    )
    assertEvalsTo(
      ToArray(streamScanIR(IRStream(1, null, 3), NA(TInt32))((accum, elt) =>
        accum + elt
      )),
      FastSeq(null, null, null, null),
    )
    assertEvalsTo(ToArray(streamScanIR(NA(TStream(TInt32)), 0)((accum, elt) => accum + elt)), null)
    assertEvalsTo(
      ToArray(streamScanIR(MakeStream(IndexedSeq(), TStream(TInt32)), 99)((accum, elt) =>
        accum + elt
      )),
      FastSeq(99),
    )
    assertEvalsTo(
      ToArray(streamScanIR(
        flatMapIR(rangeIR(5))(_ => MakeStream(IndexedSeq(), TStream(TInt32))),
        99,
      )((accum, elt) => accum + elt)),
      FastSeq(99),
    )
  }

  def makeNDArray(data: IndexedSeq[Double], shape: IndexedSeq[Long], rowMajor: IR): MakeNDArray =
    MakeNDArray(
      MakeArray(data.map(F64), TArray(TFloat64)),
      MakeTuple.ordered(shape.map(I64)),
      rowMajor,
      ErrorIDs.NO_ERROR,
    )

  def makeNDArrayRef(nd: IR, indxs: IndexedSeq[Long]): NDArrayRef =
    NDArrayRef(nd, indxs.map(I64), -1)

  val scalarRowMajor = makeNDArray(FastSeq(3.0), FastSeq(), True())
  val scalarColMajor = makeNDArray(FastSeq(3.0), FastSeq(), False())
  val vectorRowMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), True())
  val vectorColMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), False())
  val matrixRowMajor = makeNDArray(FastSeq(1.0, 2.0, 3.0, 4.0), FastSeq(2, 2), True())
  val threeTensorRowMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), True())
  val threeTensorColMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), False())
  val cubeRowMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), True())
  val cubeColMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), False())

  @Test def testNDArrayShape(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    assertEvalsTo(NDArrayShape(scalarRowMajor), Row())
    assertEvalsTo(NDArrayShape(vectorRowMajor), Row(2L))
    assertEvalsTo(NDArrayShape(cubeRowMajor), Row(3L, 3L, 3L))
  }

  @Test def testNDArrayRef(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    assertEvalsTo(makeNDArrayRef(scalarRowMajor, FastSeq()), 3.0)
    assertEvalsTo(makeNDArrayRef(scalarColMajor, FastSeq()), 3.0)

    assertEvalsTo(makeNDArrayRef(vectorRowMajor, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorColMajor, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorRowMajor, FastSeq(1)), -1.0)
    assertEvalsTo(makeNDArrayRef(vectorColMajor, FastSeq(1)), -1.0)

    val threeTensorRowMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), True())
    val threeTensorColMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), False())
    val sevenRowMajor = makeNDArrayRef(threeTensorRowMajor, FastSeq(0, 1, 2))
    val sevenColMajor = makeNDArrayRef(threeTensorColMajor, FastSeq(1, 0, 1))
    // np.arange(0, 30).reshape((2, 3, 5), order="C")[0,1,2]
    assertEvalsTo(sevenRowMajor, 7.0)
    // np.arange(0, 30).reshape((2, 3, 5), order="F")[1,0,1]
    assertEvalsTo(sevenColMajor, 7.0)

    val cubeRowMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), True())
    val cubeColMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), False())
    val centerRowMajor = makeNDArrayRef(cubeRowMajor, FastSeq(1, 1, 1))
    val centerColMajor = makeNDArrayRef(cubeColMajor, FastSeq(1, 1, 1))
    assertEvalsTo(centerRowMajor, 13.0)
    assertEvalsTo(centerColMajor, 13.0)
  }

  @Test def testNDArrayReshape(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val v = NDArrayReshape(matrixRowMajor, MakeTuple.ordered(IndexedSeq(I64(4))), ErrorIDs.NO_ERROR)
    val mat2 = NDArrayReshape(v, MakeTuple.ordered(IndexedSeq(I64(2), I64(2))), ErrorIDs.NO_ERROR)

    assertEvalsTo(makeNDArrayRef(v, FastSeq(2)), 3.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastSeq(1, 0)), 3.0)
    assertEvalsTo(makeNDArrayRef(v, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastSeq(0, 0)), 1.0)
  }

  @Test def testNDArrayConcat(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    def nds(ndData: (IndexedSeq[Int], Long, Long)*): IR =
      MakeArray(
        ndData.toArray.map { case (values, nRows, nCols) =>
          if (values == null) NA(TNDArray(TInt32, Nat(2)))
          else
            MakeNDArray(
              Literal(TArray(TInt32), values),
              Literal(TTuple(TInt64, TInt64), Row(nRows, nCols)),
              True(),
              ErrorIDs.NO_ERROR,
            )
        },
        TArray(TNDArray(TInt32, Nat(2))),
      )

    val nd1 = (
      FastSeq(
        0, 1, 2,
        3, 4, 5),
      2L,
      3L,
    )

    val rowwise = (
      FastSeq(
        6, 7, 8,
        9, 10, 11,
        12, 13, 14),
      3L,
      3L,
    )

    val colwise = (
      FastSeq(
        15,
        16,
        17,
        18,
      ),
      2L,
      2L,
    )

    val emptyRowwise = (FastSeq(), 0L, 3L)
    val emptyColwise = (FastSeq(), 2L, 0L)
    val na = (null, 0L, 0L)

    val rowwiseExpected = FastSeq(
      FastSeq(0, 1, 2),
      FastSeq(3, 4, 5),
      FastSeq(6, 7, 8),
      FastSeq(9, 10, 11),
      FastSeq(12, 13, 14),
    )
    val colwiseExpected = FastSeq(
      FastSeq(0, 1, 2, 15, 16),
      FastSeq(3, 4, 5, 17, 18),
    )

    assertNDEvals(NDArrayConcat(nds(nd1, rowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, rowwise, emptyRowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, emptyRowwise, rowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(emptyRowwise, nd1, rowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(emptyRowwise), 0), FastSeq())

    assertNDEvals(NDArrayConcat(nds(nd1, colwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, colwise, emptyColwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, emptyColwise, colwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(emptyColwise, nd1, colwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(emptyColwise), 1), FastSeq(FastSeq(), FastSeq()))

    assertNDEvals(NDArrayConcat(nds(nd1, na), 1), null)
    assertNDEvals(NDArrayConcat(nds(na, na), 1), null)
    assertNDEvals(NDArrayConcat(NA(TArray(TNDArray(TInt32, Nat(2)))), 1), null)
  }

  @Test def testNDArrayMap(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val data = 0 until 10
    val shape = FastSeq(2L, 5L)

    val positives = makeNDArray(data.map(_.toDouble), shape, True())
    val negatives = ndMap(positives)(e => -e)
    assertEvalsTo(makeNDArrayRef(positives, FastSeq(1L, 0L)), 5.0)
    assertEvalsTo(makeNDArrayRef(negatives, FastSeq(1L, 0L)), -5.0)

    val trues = MakeNDArray(
      MakeArray(data.map(_ => True()), TArray(TBoolean)),
      MakeTuple.ordered(shape.map(I64)),
      True(),
      ErrorIDs.NO_ERROR,
    )
    val falses = ndMap(trues)(e => !e)
    assertEvalsTo(makeNDArrayRef(trues, FastSeq(1L, 0L)), true)
    assertEvalsTo(makeNDArrayRef(falses, FastSeq(1L, 0L)), false)

    val bools = MakeNDArray(
      MakeArray(data.map(i => if (i % 2 == 0) True() else False()), TArray(TBoolean)),
      MakeTuple.ordered(shape.map(I64)),
      False(),
      ErrorIDs.NO_ERROR,
    )
    val boolsToBinary = ndMap(bools)(e => If(e, I64(1L), I64(0L)))
    val one = makeNDArrayRef(boolsToBinary, FastSeq(0L, 0L))
    val zero = makeNDArrayRef(boolsToBinary, FastSeq(1L, 1L))
    assertEvalsTo(one, 1L)
    assertEvalsTo(zero, 0L)
  }

  @Test def testNDArrayMap2(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val shape = MakeTuple.ordered(FastSeq(2L, 2L).map(I64))
    val numbers = MakeNDArray(
      MakeArray((0 until 4).map(i => F64(i.toDouble)), TArray(TFloat64)),
      shape,
      True(),
      ErrorIDs.NO_ERROR,
    )
    val bools = MakeNDArray(
      MakeArray(IndexedSeq(True(), False(), False(), True()), TArray(TBoolean)),
      shape,
      True(),
      ErrorIDs.NO_ERROR,
    )

    val actual = ndMap2(numbers, bools)((n, b) => n + If(b, F64(10), F64(20)))
    val ten = makeNDArrayRef(actual, FastSeq(0L, 0L))
    val twentyTwo = makeNDArrayRef(actual, FastSeq(1L, 0L))
    assertEvalsTo(ten, 10.0)
    assertEvalsTo(twentyTwo, 22.0)
  }

  @Test def testNDArrayReindex(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val transpose = NDArrayReindex(matrixRowMajor, FastSeq(1, 0))
    val identity = NDArrayReindex(matrixRowMajor, FastSeq(0, 1))

    val topLeftIndex = FastSeq(0L, 0L)
    val bottomLeftIndex = FastSeq(1L, 0L)

    assertEvalsTo(makeNDArrayRef(matrixRowMajor, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(identity, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(transpose, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(matrixRowMajor, bottomLeftIndex), 3.0)
    assertEvalsTo(makeNDArrayRef(identity, bottomLeftIndex), 3.0)
    assertEvalsTo(makeNDArrayRef(transpose, bottomLeftIndex), 2.0)

    val partialTranspose = NDArrayReindex(cubeRowMajor, FastSeq(0, 2, 1))
    val idx = FastSeq(0L, 1L, 0L)
    val partialTranposeIdx = FastSeq(0L, 0L, 1L)
    assertEvalsTo(makeNDArrayRef(cubeRowMajor, idx), 3.0)
    assertEvalsTo(makeNDArrayRef(partialTranspose, partialTranposeIdx), 3.0)
  }

  @Test def testNDArrayBroadcasting(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    def sum(nd1: IR, nd2: IR): IR = ndMap2(nd1, nd2)(_ + _)

    val scalarWithMatrix = sum(NDArrayReindex(scalarRowMajor, FastSeq(1, 0)), matrixRowMajor)

    val topLeft = makeNDArrayRef(scalarWithMatrix, FastSeq(0, 0))
    assertEvalsTo(topLeft, 4.0)

    val vectorWithMatrix = sum(NDArrayReindex(vectorRowMajor, FastSeq(1, 0)), matrixRowMajor)

    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastSeq(0, 1)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastSeq(1, 0)), 4.0)

    val colVector = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2, 1), True())
    val colVectorWithMatrix = sum(colVector, matrixRowMajor)

    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastSeq(0, 1)), 3.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastSeq(1, 0)), 2.0)

    val vectorWithEmpty = sum(
      NDArrayReindex(vectorRowMajor, FastSeq(1, 0)),
      makeNDArray(FastSeq(), FastSeq(0, 2), True()),
    )
    assertEvalsTo(NDArrayShape(vectorWithEmpty), Row(0L, 2L))

    val colVectorWithEmpty = sum(colVector, makeNDArray(FastSeq(), FastSeq(2, 0), True()))
    assertEvalsTo(NDArrayShape(colVectorWithEmpty), Row(2L, 0L))
  }

  @Test def testNDArrayAgg(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val empty = makeNDArrayRef(
      NDArrayAgg(makeNDArray(IndexedSeq(), IndexedSeq(0, 5), true), IndexedSeq(0, 1)),
      IndexedSeq(),
    )
    assertEvalsTo(empty, 0.0)

    val three = makeNDArrayRef(NDArrayAgg(scalarRowMajor, IndexedSeq.empty), IndexedSeq())
    assertEvalsTo(three, 3.0)

    val zero = makeNDArrayRef(NDArrayAgg(vectorRowMajor, IndexedSeq(0)), IndexedSeq.empty)
    assertEvalsTo(zero, 0.0)

    val four = makeNDArrayRef(NDArrayAgg(matrixRowMajor, IndexedSeq(0)), IndexedSeq(0))
    assertEvalsTo(four, 4.0)
    val six = makeNDArrayRef(NDArrayAgg(matrixRowMajor, IndexedSeq(0)), IndexedSeq(1))
    assertEvalsTo(six, 6.0)

    val twentySeven = makeNDArrayRef(NDArrayAgg(cubeRowMajor, IndexedSeq(2)), IndexedSeq(0, 0))
    assertEvalsTo(twentySeven, 3.0)
  }

  @Test def testNDArrayMatMul(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val dotProduct = NDArrayMatMul(vectorRowMajor, vectorRowMajor, ErrorIDs.NO_ERROR)
    val zero = makeNDArrayRef(dotProduct, IndexedSeq())
    assertEvalsTo(zero, 2.0)

    val seven = makeNDArrayRef(
      NDArrayMatMul(matrixRowMajor, matrixRowMajor, ErrorIDs.NO_ERROR),
      IndexedSeq(0, 0),
    )
    assertEvalsTo(seven, 7.0)

    val twoByThreeByFive = threeTensorRowMajor
    val twoByFiveByThree = NDArrayReindex(twoByThreeByFive, IndexedSeq(0, 2, 1))
    val twoByThreeByThree = NDArrayMatMul(twoByThreeByFive, twoByFiveByThree, ErrorIDs.NO_ERROR)
    val thirty = makeNDArrayRef(twoByThreeByThree, IndexedSeq(0, 0, 0))
    assertEvalsTo(thirty, 30.0)

    val threeByTwoByFive = NDArrayReindex(twoByThreeByFive, IndexedSeq(1, 0, 2))
    val matMulCube = NDArrayMatMul(
      NDArrayReindex(matrixRowMajor, IndexedSeq(2, 0, 1)),
      threeByTwoByFive,
      ErrorIDs.NO_ERROR,
    )
    assertEvalsTo(makeNDArrayRef(matMulCube, IndexedSeq(0, 0, 0)), 30.0)
  }

  @Test def testNDArrayInv(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly
    val matrixRowMajor = makeNDArray(FastSeq(1.5, 2.0, 4.0, 5.0), FastSeq(2, 2), True())
    val inv = NDArrayInv(matrixRowMajor, ErrorIDs.NO_ERROR)
    val expectedInv = FastSeq(FastSeq(-10.0, 4.0), FastSeq(8.0, -3.0))
    assertNDEvals(inv, expectedInv)
  }

  @Test def testNDArraySlice(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val rightCol = NDArraySlice(
      matrixRowMajor,
      MakeTuple.ordered(IndexedSeq(MakeTuple.ordered(IndexedSeq(I64(0), I64(2), I64(1))), I64(1))),
    )
    assertEvalsTo(NDArrayShape(rightCol), Row(2L))
    assertEvalsTo(makeNDArrayRef(rightCol, FastSeq(0)), 2.0)
    assertEvalsTo(makeNDArrayRef(rightCol, FastSeq(1)), 4.0)

    val topRow = NDArraySlice(
      matrixRowMajor,
      MakeTuple.ordered(IndexedSeq(
        I64(0),
        MakeTuple.ordered(IndexedSeq(
          I64(0),
          GetTupleElement(NDArrayShape(matrixRowMajor), 1),
          I64(1),
        )),
      )),
    )
    assertEvalsTo(makeNDArrayRef(topRow, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(topRow, FastSeq(1)), 2.0)

    val scalarSlice = NDArraySlice(scalarRowMajor, MakeTuple.ordered(FastSeq()))
    assertEvalsTo(makeNDArrayRef(scalarSlice, FastSeq()), 3.0)
  }

  @Test def testNDArrayFilter(): scalatest.Assertion = {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastSeq(NA(TArray(TInt64)), NA(TArray(TInt64)))),
      FastSeq(FastSeq(1.0, 2.0), FastSeq(3.0, 4.0)),
    )

    assertNDEvals(
      NDArrayFilter(
        matrixRowMajor,
        FastSeq(
          MakeArray(FastSeq(I64(0), I64(1)), TArray(TInt64)),
          MakeArray(FastSeq(I64(0), I64(1)), TArray(TInt64)),
        ),
      ),
      FastSeq(FastSeq(1.0, 2.0), FastSeq(3.0, 4.0)),
    )

    assertNDEvals(
      NDArrayFilter(
        matrixRowMajor,
        FastSeq(
          MakeArray(FastSeq(I64(1), I64(0)), TArray(TInt64)),
          MakeArray(FastSeq(I64(1), I64(0)), TArray(TInt64)),
        ),
      ),
      FastSeq(FastSeq(4.0, 3.0), FastSeq(2.0, 1.0)),
    )

    assertNDEvals(
      NDArrayFilter(
        matrixRowMajor,
        FastSeq(
          MakeArray(FastSeq(I64(0)), TArray(TInt64)),
          NA(TArray(TInt64)),
        ),
      ),
      FastSeq(FastSeq(1.0, 2.0)),
    )

    assertNDEvals(
      NDArrayFilter(
        matrixRowMajor,
        FastSeq(
          NA(TArray(TInt64)),
          MakeArray(FastSeq(I64(0)), TArray(TInt64)),
        ),
      ),
      FastSeq(FastSeq(1.0), FastSeq(3.0)),
    )

    assertNDEvals(
      NDArrayFilter(
        matrixRowMajor,
        FastSeq(
          MakeArray(FastSeq(I64(1)), TArray(TInt64)),
          MakeArray(FastSeq(I64(1)), TArray(TInt64)),
        ),
      ),
      FastSeq(FastSeq(4.0)),
    )
  }

  private def join(
    left: IR,
    right: IR,
    lKeys: IndexedSeq[String],
    rKeys: IndexedSeq[String],
    rightDistinct: Boolean,
    joinType: String,
  ): IR = {
    ToArray(joinIR(
      left,
      right,
      lKeys,
      rKeys,
      joinType,
      requiresMemoryManagement = false,
      rightKeyIsDistinct = rightDistinct,
    ) { (l: IR, r: IR) =>
      bindIRs(l, r) { case Seq(l, r) =>
        MakeStruct(
          (lKeys, rKeys).zipped.map { (lk, rk) =>
            lk -> Coalesce(IndexedSeq(GetField(l, lk), GetField(r, rk)))
          }
            ++ tcoerce[TStruct](l.typ).fields.filter(f => !lKeys.contains(f.name)).map { f =>
              f.name -> GetField(l, f.name)
            } ++ tcoerce[TStruct](r.typ).fields.filter(f => !rKeys.contains(f.name)).map { f =>
              f.name -> GetField(r, f.name)
            }
        )
      }
    })
  }

  @Test def testStreamZipJoin(): scalatest.Assertion = {
    def eltType = TStruct("k1" -> TInt32, "k2" -> TString, "idx" -> TInt32)
    def makeStream(a: IndexedSeq[Integer]): IR = {
      if (a == null)
        NA(TStream(eltType))
      else
        MakeStream(
          a.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "k1" -> (if (n == null) NA(TInt32) else I32(n)),
              "k2" -> Str("x"),
              "idx" -> I32(idx),
            ))
          },
          TStream(eltType),
        )
    }

    def zipJoin(as: IndexedSeq[IndexedSeq[Integer]], key: Int): IR = {
      val streams = as.map(makeStream)
      ToArray(zipJoin2IR(
        streams,
        FastSeq("k1", "k2").take(key),
      )((curKey, curVals) =>
        InsertFields(curKey, FastSeq("vals" -> curVals))
      ))
    }

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](0, 1, null), null), 1),
      null,
    )

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](0, 1, null), Array[Integer](1, 2, null)), 1),
      FastSeq(
        Row(0, FastSeq(Row(0, "x", 0), null)),
        Row(1, FastSeq(Row(1, "x", 1), Row(1, "x", 0))),
        Row(2, FastSeq(null, Row(2, "x", 1))),
        Row(null, FastSeq(Row(null, "x", 2), null)),
        Row(null, FastSeq(null, Row(null, "x", 2))),
      ),
    )

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](0, 1), Array[Integer](1, 2), Array[Integer](0, 2)), 1),
      FastSeq(
        Row(0, FastSeq(Row(0, "x", 0), null, Row(0, "x", 0))),
        Row(1, FastSeq(Row(1, "x", 1), Row(1, "x", 0), null)),
        Row(2, FastSeq(null, Row(2, "x", 1), Row(2, "x", 1))),
      ),
    )

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](0, 1), Array[Integer](), Array[Integer](0, 2)), 1),
      FastSeq(
        Row(0, FastSeq(Row(0, "x", 0), null, Row(0, "x", 0))),
        Row(1, FastSeq(Row(1, "x", 1), null, null)),
        Row(2, FastSeq(null, null, Row(2, "x", 1))),
      ),
    )

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](), Array[Integer]()), 1),
      FastSeq(),
    )

    assertEvalsTo(
      zipJoin(FastSeq(Array[Integer](0, 1)), 1),
      FastSeq(
        Row(0, FastSeq(Row(0, "x", 0))),
        Row(1, FastSeq(Row(1, "x", 1))),
      ),
    )
  }

  @Test def testStreamMultiMerge(): scalatest.Assertion = {
    def eltType = TStruct("k1" -> TInt32, "k2" -> TString, "idx" -> TInt32)
    def makeStream(a: IndexedSeq[Integer]): IR = {
      if (a == null)
        NA(TStream(eltType))
      else
        MakeStream(
          a.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "k1" -> (if (n == null) NA(TInt32) else I32(n)),
              "k2" -> Str("x"),
              "idx" -> I32(idx),
            ))
          },
          TStream(eltType),
        )
    }

    def merge(as: IndexedSeq[IndexedSeq[Integer]], key: Int): IR = {
      val streams = as.map(makeStream)
      ToArray(StreamMultiMerge(streams, FastSeq("k1", "k2").take(key)))
    }

    // TODO: add more iterator compilation infrastructure so that multimerge can be strict again
    // assertEvalsTo(
    //   merge(FastIndexedSeq(Array[Integer](0, 1, null, null), null), 1),
    //   null)

    assertEvalsTo(
      merge(FastSeq(Array[Integer](0, 1, null, null), Array[Integer](1, 2, null, null)), 1),
      FastSeq(
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(1, "x", 0),
        Row(2, "x", 1),
        Row(null, "x", 2),
        Row(null, "x", 3),
        Row(null, "x", 2),
        Row(null, "x", 3),
      ),
    )

    assertEvalsTo(
      merge(FastSeq(Array[Integer](0, 1), Array[Integer](1, 2), Array[Integer](0, 2)), 1),
      FastSeq(
        Row(0, "x", 0),
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(1, "x", 0),
        Row(2, "x", 1),
        Row(2, "x", 1),
      ),
    )

    assertEvalsTo(
      merge(FastSeq(Array[Integer](0, 1), Array[Integer](), Array[Integer](0, 2)), 1),
      FastSeq(
        Row(0, "x", 0),
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(2, "x", 1),
      ),
    )

    assertEvalsTo(
      merge(FastSeq(Array[Integer](), Array[Integer]()), 1),
      FastSeq(),
    )

    assertEvalsTo(
      merge(FastSeq(Array[Integer](0, 1)), 1),
      FastSeq(
        Row(0, "x", 0),
        Row(1, "x", 1),
      ),
    )
  }

  @Test def testJoinRightDistinct(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    def joinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], joinType: String): IR = {
      join(
        MakeStream.unify(
          ctx,
          left.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "lk1" -> (if (n == null) NA(TInt32) else I32(n)),
              "lk2" -> Str("x"),
              "a" -> I64(idx),
            ))
          },
        ),
        MakeStream.unify(
          ctx,
          right.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "b" -> I32(idx),
              "rk2" -> Str("x"),
              "rk1" -> (if (n == null) NA(TInt32) else I32(n)),
              "c" -> Str("foo"),
            ))
          },
        ),
        FastSeq("lk1", "lk2"),
        FastSeq("rk1", "rk2"),
        rightDistinct = true,
        joinType,
      )
    }
    def leftJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "left")
    def outerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "outer")

    assertEvalsTo(
      join(
        NA(TStream(TStruct("k1" -> TInt32, "k2" -> TString, "a" -> TInt64))),
        MakeStream.unify(
          ctx,
          IndexedSeq(MakeStruct(FastSeq(
            "b" -> I32(0),
            "k2" -> Str("x"),
            "k1" -> I32(3),
            "c" -> Str("foo"),
          ))),
        ),
        FastSeq("k1", "k2"),
        FastSeq("k1", "k2"),
        true,
        "left",
      ),
      null,
    )

    assertEvalsTo(
      join(
        MakeStream.unify(
          ctx,
          IndexedSeq(MakeStruct(FastSeq("k1" -> I32(0), "k2" -> Str("x"), "a" -> I64(3)))),
        ),
        NA(TStream(TStruct("b" -> TInt32, "k2" -> TString, "k1" -> TInt32, "c" -> TString))),
        FastSeq("k1", "k2"),
        FastSeq("k1", "k2"),
        true,
        "left",
      ),
      null,
    )

    assertEvalsTo(
      leftJoinRows(Array[Integer](0, null), Array[Integer](1, null)),
      FastSeq(
        Row(0, "x", 0L, null, null),
        Row(null, "x", 1L, null, null),
      ),
    )

    assertEvalsTo(
      outerJoinRows(Array[Integer](0, null), Array[Integer](1, null)),
      FastSeq(
        Row(0, "x", 0L, null, null),
        Row(1, "x", null, 0, "foo"),
        Row(null, "x", 1L, null, null),
        Row(null, "x", null, 1, "foo"),
      ),
    )

    assertEvalsTo(
      leftJoinRows(Array[Integer](0, 1, 2), Array[Integer](1)),
      FastSeq(
        Row(0, "x", 0L, null, null),
        Row(1, "x", 1L, 0, "foo"),
        Row(2, "x", 2L, null, null),
      ),
    )

    assertEvalsTo(
      leftJoinRows(Array[Integer](0, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)),
      FastSeq(
        Row(0, "x", 0L, 1, "foo"),
        Row(1, "x", 1L, 3, "foo"),
        Row(2, "x", 2L, 5, "foo"),
      ),
    )

    assertEvalsTo(
      leftJoinRows(Array[Integer](0, 1, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)),
      FastSeq(
        Row(0, "x", 0L, 1, "foo"),
        Row(1, "x", 1L, 3, "foo"),
        Row(1, "x", 2L, 3, "foo"),
        Row(2, "x", 3L, 5, "foo"),
      ),
    )
  }

  @Test def testStreamJoin(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    def joinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], joinType: String): IR = {
      join(
        MakeStream.unify(
          ctx,
          left.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq("lk" -> (if (n == null) NA(TInt32) else I32(n)), "l" -> I32(idx)))
          },
        ),
        MakeStream.unify(
          ctx,
          right.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq("rk" -> (if (n == null) NA(TInt32) else I32(n)), "r" -> I32(idx)))
          },
        ),
        FastSeq("lk"),
        FastSeq("rk"),
        false,
        joinType,
      )
    }
    def leftJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "left")
    def outerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "outer")
    def innerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "inner")
    def rightJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "right")

    assertEvalsTo(
      leftJoinRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
      ),
      FastSeq(
        Row(1, 0, 2),
        Row(1, 0, 3),
        Row(1, 1, 2),
        Row(1, 1, 3),
        Row(2, 2, null),
        Row(2, 3, null),
        Row(null, 4, null),
        Row(null, 5, null),
      ),
    )

    assertEvalsTo(
      outerJoinRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
      ),
      FastSeq(
        Row(0, null, 0),
        Row(0, null, 1),
        Row(1, 0, 2),
        Row(1, 0, 3),
        Row(1, 1, 2),
        Row(1, 1, 3),
        Row(2, 2, null),
        Row(2, 3, null),
        Row(3, null, 4),
        Row(3, null, 5),
        Row(null, 4, null),
        Row(null, 5, null),
        Row(null, null, 6),
        Row(null, null, 7),
      ),
    )

    assertEvalsTo(
      innerJoinRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
      ),
      FastSeq(
        Row(1, 0, 2),
        Row(1, 0, 3),
        Row(1, 1, 2),
        Row(1, 1, 3),
      ),
    )

    assertEvalsTo(
      rightJoinRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
      ),
      FastSeq(
        Row(0, null, 0),
        Row(0, null, 1),
        Row(1, 0, 2),
        Row(1, 0, 3),
        Row(1, 1, 2),
        Row(1, 1, 3),
        Row(3, null, 4),
        Row(3, null, 5),
        Row(null, null, 6),
        Row(null, null, 7),
      ),
    )
  }

  @Test def testStreamMerge(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    def mergeRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], key: Int): IR = {
      val typ = TStream(TStruct("k" -> TInt32, "sign" -> TInt32, "idx" -> TInt32))
      ToArray(StreamMultiMerge(
        FastSeq(
          if (left == null)
            NA(typ)
          else
            MakeStream(
              left.zipWithIndex.map { case (n, idx) =>
                MakeStruct(FastSeq(
                  "k" -> (if (n == null) NA(TInt32) else I32(n)),
                  "sign" -> I32(1),
                  "idx" -> I32(idx),
                ))
              },
              typ,
            ),
          if (right == null)
            NA(typ)
          else
            MakeStream(
              right.zipWithIndex.map { case (n, idx) =>
                MakeStruct(FastSeq(
                  "k" -> (if (n == null) NA(TInt32) else I32(n)),
                  "sign" -> I32(-1),
                  "idx" -> I32(idx),
                ))
              },
              typ,
            ),
        ),
        FastSeq("k", "sign").take(key),
      ))
    }

    assertEvalsTo(
      mergeRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
        1,
      ),
      FastSeq(
        Row(0, -1, 0),
        Row(0, -1, 1),
        Row(1, 1, 0),
        Row(1, 1, 1),
        Row(1, -1, 2),
        Row(1, -1, 3),
        Row(2, 1, 2),
        Row(2, 1, 3),
        Row(3, -1, 4),
        Row(3, -1, 5),
        Row(null, 1, 4),
        Row(null, 1, 5),
        Row(null, -1, 6),
        Row(null, -1, 7),
      ),
    )

    // right stream ends first
    assertEvalsTo(
      mergeRows(Array[Integer](1, 1, 2, 2), Array[Integer](0, 0, 1, 1), 1),
      FastSeq(
        Row(0, -1, 0),
        Row(0, -1, 1),
        Row(1, 1, 0),
        Row(1, 1, 1),
        Row(1, -1, 2),
        Row(1, -1, 3),
        Row(2, 1, 2),
        Row(2, 1, 3),
      ),
    )

    // compare on two key fields
    assertEvalsTo(
      mergeRows(
        Array[Integer](1, 1, 2, 2, null, null),
        Array[Integer](0, 0, 1, 1, 3, 3, null, null),
        2,
      ),
      FastSeq(
        Row(0, -1, 0),
        Row(0, -1, 1),
        Row(1, -1, 2),
        Row(1, -1, 3),
        Row(1, 1, 0),
        Row(1, 1, 1),
        Row(2, 1, 2),
        Row(2, 1, 3),
        Row(3, -1, 4),
        Row(3, -1, 5),
        Row(null, 1, 4),
        Row(null, 1, 5),
        Row(null, -1, 6),
        Row(null, -1, 7),
      ),
    )

    // right stream empty
    assertEvalsTo(
      mergeRows(Array[Integer](1, 2, null), Array[Integer](), 1),
      FastSeq(
        Row(1, 1, 0),
        Row(2, 1, 1),
        Row(null, 1, 2),
      ),
    )

    // left stream empty
    assertEvalsTo(
      mergeRows(Array[Integer](), Array[Integer](1, 2, null), 1),
      FastSeq(
        Row(1, -1, 0),
        Row(2, -1, 1),
        Row(null, -1, 2),
      ),
    )

    // one stream missing
    assertEvalsTo(mergeRows(null, Array[Integer](1, 2, null), 1), null)
    assertEvalsTo(mergeRows(Array[Integer](1, 2, null), null, 1), null)
  }

  @Test def testDie(): scalatest.Assertion = {
    assertFatal(Die("mumblefoo", TFloat64), "mble")
    assertFatal(Die(NA(TString), TFloat64, -1), "message missing")
  }

  @Test def testStreamRange(): scalatest.Assertion = {
    def assertEquals(start: Integer, stop: Integer, step: Integer, expected: IndexedSeq[Int])
      : scalatest.Assertion =
      assertEvalsTo(
        ToArray(StreamRange(In(0, TInt32), In(1, TInt32), In(2, TInt32))),
        args = FastSeq(start -> TInt32, stop -> TInt32, step -> TInt32),
        expected = expected,
      )
    assertEquals(0, 5, null, null)
    assertEquals(0, null, 1, null)
    assertEquals(null, 5, 1, null)

    assertFatal(ToArray(StreamRange(I32(0), I32(5), I32(0))), "step size")

    forAll(
      for {
        start <- -2 to 2
        stop <- -2 to 8
        step <- 1 to 3
      } yield (start, stop, step)
    ) { case (start, stop, step) =>
      assertEquals(start, stop, step, expected = Array.range(start, stop, step).toFastSeq)
      assertEquals(start, stop, -step, expected = Array.range(start, stop, -step).toFastSeq)
    }
    // this needs to be written this way because of a bug in Scala's Array.range
    val expected = Array.tabulate(11)(Int.MinValue + _ * (Int.MaxValue / 5)).toFastSeq
    assertEquals(Int.MinValue, Int.MaxValue, Int.MaxValue / 5, expected)
  }

  @Test def testArrayAgg(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    val sumSig = AggSignature(Sum(), IndexedSeq(), IndexedSeq(TInt64))
    assertEvalsTo(
      streamAggIR(mapIR(rangeIR(4))(x => Cast(x, TInt64))) { x =>
        ApplyAggOp(FastSeq.empty, FastSeq(x), sumSig)
      },
      6L,
    )
  }

  @Test def testArrayAggContexts(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    val ir = bindIR(In(0, TInt32) * In(0, TInt32)) { x => // multiply to prevent forwarding
      streamAggIR(rangeIR(10)) { elt =>
        aggBindIR(Cast(x * x * elt, TInt64)) { y => // different type to trigger validation errors
          invoke(
            "append",
            TArray(TArray(TInt32)),
            ApplyAggOp(
              FastSeq(),
              FastSeq(
                MakeArray(
                  FastSeq(
                    x,
                    elt,
                    Cast(y, TInt32),
                    Cast(y, TInt32),
                  ), // reference y twice to prevent forwarding
                  TArray(TInt32),
                )
              ),
              AggSignature(Collect(), FastSeq(), FastSeq(TArray(TInt32))),
            ),
            MakeArray(FastSeq(x), TArray(TInt32)),
          )
        }
      }
    }

    assertEvalsTo(
      ir,
      FastSeq(1 -> TInt32),
      (0 until 10).map(i => FastSeq(1, i, i, i)) ++ FastSeq(FastSeq(1)),
    )
  }

  @Test def testStreamAggScan(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly

    val eltType = TStruct("x" -> TCall, "y" -> TInt32)

    val ir = streamAggScanIR(ToStream(In(0, TArray(eltType)))) { foo =>
      GetField(foo, "y") +
        GetField(
          ApplyScanOp(
            FastSeq(I32(2)),
            FastSeq(GetField(foo, "x")),
            AggSignature(CallStats(), FastSeq(TInt32), FastSeq(TCall)),
          ),
          "AN",
        )
    }

    val input = FastSeq(
      Row(null, 1),
      Row(Call2(0, 0), 2),
      Row(Call2(0, 1), 3),
      Row(Call2(1, 1), 4),
      null,
      Row(null, 5),
    ) -> TArray(eltType)

    assertEvalsTo(
      ToArray(ir),
      args = FastSeq(input),
      expected = FastSeq(1 + 0, 2 + 0, 3 + 2, 4 + 4, null, 5 + 6),
    )

    assertEvalsTo(StreamLen(ir), args = FastSeq(input), 6)
  }

  @Test def testInsertFields(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val s = TStruct("a" -> TInt64, "b" -> TString)
    val emptyStruct = MakeStruct(IndexedSeq("a" -> NA(TInt64), "b" -> NA(TString)))

    assertEvalsTo(
      InsertFields(
        NA(s),
        IndexedSeq(),
      ),
      null,
    )

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        IndexedSeq("a" -> I64(5)),
      ),
      Row(5L, null),
    )

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        IndexedSeq("c" -> F64(3.2)),
      ),
      Row(null, null, 3.2),
    )

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        IndexedSeq("c" -> NA(TFloat64)),
      ),
      Row(null, null, null),
    )

    assertEvalsTo(
      InsertFields(
        MakeStruct(IndexedSeq("a" -> NA(TInt64), "b" -> Str("abc"))),
        IndexedSeq(),
      ),
      Row(null, "abc"),
    )

    assertEvalsTo(
      InsertFields(
        MakeStruct(IndexedSeq("a" -> NA(TInt64), "b" -> Str("abc"))),
        IndexedSeq("a" -> I64(5)),
      ),
      Row(5L, "abc"),
    )

    assertEvalsTo(
      InsertFields(
        MakeStruct(IndexedSeq("a" -> NA(TInt64), "b" -> Str("abc"))),
        IndexedSeq("c" -> F64(3.2)),
      ),
      Row(null, "abc", 3.2),
    )

    assertEvalsTo(
      InsertFields(NA(TStruct("a" -> TInt32)), IndexedSeq("foo" -> I32(5))),
      null,
    )

    assertEvalsTo(
      InsertFields(
        In(0, s),
        IndexedSeq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastSeq("c", "d", "e", "a", "b")),
      ),
      FastSeq(Row(null, "abc") -> s),
      Row(3.2, 5.5, 6.6, null, "abc"),
    )

    assertEvalsTo(
      InsertFields(
        In(0, s),
        IndexedSeq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastSeq("a", "b", "c", "d", "e")),
      ),
      FastSeq(Row(null, "abc") -> s),
      Row(null, "abc", 3.2, 5.5, 6.6),
    )

    assertEvalsTo(
      InsertFields(
        In(0, s),
        IndexedSeq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastSeq("c", "a", "d", "b", "e")),
      ),
      FastSeq(Row(null, "abc") -> s),
      Row(3.2, null, 5.5, "abc", 6.6),
    )

  }

  @Test def testSelectFields(): scalatest.Assertion = {
    assertEvalsTo(
      SelectFields(
        NA(TStruct("foo" -> TInt32, "bar" -> TFloat64)),
        FastSeq("foo"),
      ),
      null,
    )

    assertEvalsTo(
      SelectFields(
        MakeStruct(FastSeq("foo" -> 6, "bar" -> 0.0)),
        FastSeq("foo"),
      ),
      Row(6),
    )

    assertEvalsTo(
      SelectFields(
        MakeStruct(FastSeq("a" -> 6, "b" -> 0.0, "c" -> 3L)),
        FastSeq("b", "a"),
      ),
      Row(0.0, 6),
    )
  }

  @Test def testGetField(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    val s = MakeStruct(IndexedSeq("a" -> NA(TInt64), "b" -> Str("abc")))
    val na = NA(TStruct("a" -> TInt64, "b" -> TString))

    assertEvalsTo(GetField(s, "a"), null)
    assertEvalsTo(GetField(s, "b"), "abc")
    assertEvalsTo(GetField(na, "a"), null)
  }

  @Test def testLiteral(): scalatest.Assertion = {
    implicit val execStrats =
      Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.JvmCompile)
    val poopEmoji = new String(Array[Char](0xd83d, 0xdca9))
    val types = Array(
      TTuple(TInt32, TString, TArray(TInt32)),
      TArray(TString),
      TDict(TInt32, TString),
    )
    val values = Array(
      Row(400, "foo" + poopEmoji, FastSeq(4, 6, 8)),
      FastSeq(poopEmoji, "", "foo"),
      Map[Int, String](1 -> "", 5 -> "foo", -4 -> poopEmoji),
    )

    assertEvalsTo(Literal(types(0), values(0)), values(0))
    assertEvalsTo(
      MakeTuple.ordered(types.zip(values).map { case (t, v) => Literal(t, v) }),
      Row.fromSeq(values.toFastSeq),
    )
    assertEvalsTo(Str("hello" + poopEmoji), "hello" + poopEmoji)
  }

  @Test def testSameLiteralsWithDifferentTypes(): scalatest.Assertion = {
    assertEvalsTo(
      ApplyComparisonOp(
        EQ(TArray(TInt32)),
        ToArray(
          mapIR(ToStream(Literal(TArray(TFloat64), FastSeq(1.0, 2.0))))(Cast(_, TInt32))
        ),
        Literal(TArray(TInt32), FastSeq(1, 2)),
      ),
      true,
    )
  }

  @Test def testTableCount(): scalatest.Assertion = {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    assertEvalsTo(TableCount(TableRange(0, 4)), 0L)
    assertEvalsTo(TableCount(TableRange(7, 4)), 7L)
  }

  @Test def testTableGetGlobals(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    assertEvalsTo(
      TableGetGlobals(TableMapGlobals(TableRange(0, 1), Literal(TStruct("a" -> TInt32), Row(1)))),
      Row(1),
    )
  }

  @Test def testTableAggregate(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.allRelational

    val table = TableRange(3, 2)
    val countSig = AggSignature(Count(), IndexedSeq(), IndexedSeq())
    val count = ApplyAggOp(FastSeq.empty, FastSeq.empty, countSig)
    assertEvalsTo(TableAggregate(table, MakeStruct(IndexedSeq("foo" -> count))), Row(3L))
  }

  @Test def testMatrixAggregate(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly

    val matrix = MatrixIR.range(5, 5, None)
    val countSig = AggSignature(Count(), IndexedSeq(), IndexedSeq())
    val count = ApplyAggOp(FastSeq.empty, FastSeq.empty, countSig)
    assertEvalsTo(MatrixAggregate(matrix, MakeStruct(IndexedSeq("foo" -> count))), Row(25L))
  }

  @Test def testGroupByKey(): scalatest.Assertion = {
    implicit val execStrats = Set(
      ExecStrategy.Interpret,
      ExecStrategy.InterpretUnoptimized,
      ExecStrategy.JvmCompile,
      ExecStrategy.JvmCompileUnoptimized,
    )

    def tuple(k: String, v: Int): IR = MakeTuple.ordered(IndexedSeq(Str(k), I32(v)))

    def groupby(tuples: IR*): IR =
      GroupByKey(MakeStream(tuples.toArray[IR], TStream(TTuple(TString, TInt32))))

    val collection1 = groupby(
      tuple("foo", 0),
      tuple("bar", 4),
      tuple("foo", -1),
      tuple("bar", 0),
      tuple("foo", 10),
      tuple("", 0),
    )
    assertEvalsTo(
      collection1,
      Map("" -> FastSeq(0), "bar" -> FastSeq(4, 0), "foo" -> FastSeq(0, -1, 10)),
    )

    assertEvalsTo(groupby(), Map())
  }

  @DataProvider(name = "compareDifferentTypes")
  def compareDifferentTypesData(): Array[Array[Any]] = Array(
    Array(FastSeq(0.0, 0.0), TArray(TFloat64), TArray(TFloat64)),
    Array(Set(0, 1), TSet(TInt32), TSet(TInt32)),
    Array(Map(0L -> 5, 3L -> 20), TDict(TInt64, TInt32), TDict(TInt64, TInt32)),
    Array(
      Interval(1, 2, includesStart = false, includesEnd = true),
      TInterval(TInt32),
      TInterval(TInt32),
    ),
    Array(
      Row("foo", 0.0),
      TStruct("a" -> TString, "b" -> TFloat64),
      TStruct("a" -> TString, "b" -> TFloat64),
    ),
    Array(Row("foo", 0.0), TTuple(TString, TFloat64), TTuple(TString, TFloat64)),
    Array(
      Row(FastSeq("foo"), 0.0),
      TTuple(TArray(TString), TFloat64),
      TTuple(TArray(TString), TFloat64),
    ),
  )

  @Test(dataProvider = "compareDifferentTypes")
  def testComparisonOpDifferentTypes(a: Any, t1: Type, t2: Type): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(
      ApplyComparisonOp(EQ(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      true,
    )
    assertEvalsTo(
      ApplyComparisonOp(LT(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      false,
    )
    assertEvalsTo(
      ApplyComparisonOp(GT(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      false,
    )
    assertEvalsTo(
      ApplyComparisonOp(LTEQ(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      true,
    )
    assertEvalsTo(
      ApplyComparisonOp(GTEQ(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      true,
    )
    assertEvalsTo(
      ApplyComparisonOp(NEQ(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      false,
    )
    assertEvalsTo(
      ApplyComparisonOp(EQWithNA(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      true,
    )
    assertEvalsTo(
      ApplyComparisonOp(NEQWithNA(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      false,
    )
    assertEvalsTo(
      ApplyComparisonOp(Compare(t1, t2), In(0, t1), In(1, t2)),
      FastSeq(a -> t1, a -> t2),
      0,
    )
  }

  @DataProvider(name = "valueIRs")
  def valueIRs(): Array[Array[Object]] =
    valueIRs(ctx)

  def valueIRs(ctx: ExecuteContext): Array[Array[Object]] = {
    val fs = ctx.fs

    CompileAndEvaluate(
      ctx,
      invoke(
        "index_bgen",
        TInt64,
        Array[Type](TLocus("GRCh37")),
        Str(getTestResource("example.8bits.bgen")),
        Str(getTestResource("example.8bits.bgen.idx2")),
        Literal(TDict(TString, TString), Map("01" -> "1")),
        False(),
        I32(1000000),
      ),
    )

    val b = True()
    val bin = Ref(freshName(), TBinary)
    val c = Ref(freshName(), TBoolean)
    val i = I32(5)
    val j = I32(7)
    val str = Str("Hail")
    val a = Ref(freshName(), TArray(TInt32))
    val st = Ref(freshName(), TStream(TInt32))
    val whitenStream = Ref(
      freshName(),
      TStream(TStruct(
        "prevWindow" -> TNDArray(TFloat64, Nat(2)),
        "newChunk" -> TNDArray(TFloat64, Nat(2)),
      )),
    )
    val sta = Ref(freshName(), TStream(TArray(TInt32)))
    val std = Ref(freshName(), TStream(TTuple(TInt32, TString)))
    val s = Ref(freshName(), TStruct("x" -> TInt32, "y" -> TInt64, "z" -> TFloat64))
    val t = Ref(freshName(), TTuple(TInt32, TInt64, TFloat64))
    val l = Ref(freshName(), TInt32)
    val r = Ref(freshName(), TInt32)

    val call = Ref(freshName(), TCall)

    val collectSig = AggSignature(Collect(), IndexedSeq(), IndexedSeq(TInt32))
    val pCollectSig = PhysicalAggSig(Collect(), CollectStateSig(VirtualTypeWithReq(PInt32())))

    val sumSig = AggSignature(Sum(), IndexedSeq(), IndexedSeq(TInt64))
    val pSumSig = PhysicalAggSig(Sum(), TypedStateSig(VirtualTypeWithReq(PInt64(true))))

    val callStatsSig = AggSignature(CallStats(), IndexedSeq(TInt32), IndexedSeq(TCall))
    val pCallStatsSig = PhysicalAggSig(CallStats(), CallStatsStateSig())

    val takeBySig = AggSignature(TakeBy(), IndexedSeq(TInt32), IndexedSeq(TFloat64, TInt32))

    val countSig = AggSignature(Count(), IndexedSeq(), IndexedSeq())
    val count = ApplyAggOp(FastSeq.empty, FastSeq.empty, countSig)

    val groupSignature = GroupedAggSig(VirtualTypeWithReq(PInt32(true)), FastSeq(pSumSig))

    val table = TableRange(100, 10)

    val mt = MatrixIR.range(20, 2, Some(3))
    val vcf = importVCF(ctx, getTestResource("sample.vcf"))
    val bgenReader = MatrixBGENReader(
      ctx,
      FastSeq(getTestResource("example.8bits.bgen")),
      None,
      Map.empty[String, String],
      None,
      None,
      None,
    )
    val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

    val blockMatrix =
      BlockMatrixRead(BlockMatrixNativeReader(fs, getTestResource("blockmatrix_example/0")))
    val blockMatrixWriter = BlockMatrixNativeWriter("/path/to/file.bm", false, false, false)
    val blockMatrixMultiWriter = BlockMatrixBinaryMultiWriter("/path/to/prefix", false)
    val nd = MakeNDArray(
      MakeArray(FastSeq(I32(-1), I32(1)), TArray(TInt32)),
      MakeTuple.ordered(FastSeq(I64(1), I64(2))),
      True(),
      ErrorIDs.NO_ERROR,
    )
    val rngState = RNGStateLiteral()

    def collect(ir: IR): IR =
      ApplyAggOp(FastSeq.empty, FastSeq(ir), collectSig)

    implicit def addEnv(ir: IR): (IR, BindingEnv[Type] => BindingEnv[Type]) =
      (ir, env => env)
    implicit def liftRefs(refs: Array[Ref]): BindingEnv[Type] => BindingEnv[Type] =
      env => env.bindEval(refs.map(r => r.name -> r.typ): _*)

    val irs = Array[(IR, BindingEnv[Type] => BindingEnv[Type])](
      i,
      I64(5),
      F32(3.14f),
      F64(3.14),
      str,
      True(),
      False(),
      Void(),
      UUID4(),
      Cast(i, TFloat64),
      CastRename(NA(TStruct("a" -> TInt32)), TStruct("b" -> TInt32)),
      NA(TInt32),
      IsNA(i),
      If(b, i, j),
      Switch(i, j, 0 until 7 map I32),
      Coalesce(FastSeq(i, I32(1))),
      bindIR(i)(v => v),
      aggBindIR(i)(collect(_)) -> (_.createAgg), {
        val x = Ref(freshName(), TInt32)
        x -> (_.bindEval(x.name, x.typ))
      },
      ApplyBinaryPrimOp(Add(), i, j),
      ApplyUnaryPrimOp(Negate, i),
      ApplyComparisonOp(EQ(TInt32), i, j),
      MakeArray(FastSeq(i, NA(TInt32), I32(-3)), TArray(TInt32)),
      MakeStream(FastSeq(i, NA(TInt32), I32(-3)), TStream(TInt32)),
      nd,
      NDArrayReshape(nd, MakeTuple.ordered(IndexedSeq(I64(4))), ErrorIDs.NO_ERROR),
      NDArrayConcat(MakeArray(FastSeq(nd, nd), TArray(nd.typ)), 0),
      NDArrayRef(nd, FastSeq(I64(1), I64(2)), -1),
      ndMap(nd)(v => -v),
      ndMap2(nd, nd)(_ + _),
      NDArrayReindex(nd, FastSeq(0, 1)),
      NDArrayAgg(nd, FastSeq(0)),
      NDArrayWrite(nd, Str("/path/to/ndarray")),
      NDArrayMatMul(nd, nd, ErrorIDs.NO_ERROR),
      NDArraySlice(
        nd,
        MakeTuple.ordered(FastSeq(
          MakeTuple.ordered(FastSeq(I64(0), I64(2), I64(1))),
          MakeTuple.ordered(FastSeq(I64(0), I64(2), I64(1))),
        )),
      ),
      NDArrayFilter(nd, FastSeq(NA(TArray(TInt64)), NA(TArray(TInt64)))),
      ArrayRef(a, i) -> Array(a),
      ArrayLen(a) -> Array(a),
      RNGSplit(rngState, MakeTuple.ordered(FastSeq(I64(1), I64(2), I64(3)))),
      StreamLen(st) -> Array(st),
      StreamRange(I32(0), I32(5), I32(1)),
      StreamRange(I32(0), I32(5), I32(1)),
      ArraySort(st, b) -> Array(st),
      ToSet(st) -> Array(st),
      ToDict(std) -> Array(std),
      ToArray(st) -> Array(st),
      CastToArray(NA(TSet(TInt32))),
      ToStream(a) -> Array(a),
      LowerBoundOnOrderedCollection(a, i, onKey = false) -> Array(a),
      GroupByKey(std) -> Array(std),
      StreamTake(st, I32(10)) -> Array(st),
      StreamDrop(st, I32(10)) -> Array(st),
      takeWhile(st)(_ < 5) -> Array(st),
      dropWhile(st)(_ < 5) -> Array(st),
      mapIR(st)(v => v) -> Array(st),
      zipIR(FastSeq(st, st), ArrayZipBehavior.TakeMinLength)(_ => True()) -> Array(st),
      filterIR(st)(_ => b) -> Array(st),
      flatMapIR(sta)(ToStream(_)) -> Array(sta),
      foldIR(st, I32(0))((_, v) => v) -> Array(st),
      StreamFold2(foldIR(st, I32(0))((_, v) => v)) -> Array(st),
      streamScanIR(st, I32(0))((_, v) => v) -> Array(st),
      StreamWhiten(whitenStream, "newChunk", "prevWindow", 1, 1, 1, 1, false) -> Array(
        whitenStream
      ),
      joinRightDistinctIR(
        mapIR(rangeIR(2))(x => MakeStruct(FastSeq("x" -> x))),
        mapIR(rangeIR(3))(x => MakeStruct(FastSeq("x" -> x))),
        FastSeq("x"),
        FastSeq("x"),
        "left",
      )((_, _) => I32(1)), {
        val left = mapIR(rangeIR(2))(x => MakeStruct(FastSeq("x" -> x)))
        val right = ToStream(Literal(
          TArray(TStruct("a" -> TInterval(TInt32))),
          FastSeq(Row(Interval(IntervalEndpoint(0, -1), IntervalEndpoint(1, 1)))),
        ))
        val lref = Ref(freshName(), elementType(left.typ))
        val rref = Ref(freshName(), TArray(elementType(right.typ)))
        StreamLeftIntervalJoin(
          left,
          right,
          "x",
          "a",
          lref.name,
          rref.name,
          InsertFields(lref, FastSeq("join" -> rref)),
        )
      },
      forIR(st)(_ => Void()) -> Array(st),
      streamAggIR(st)(x => ApplyAggOp(FastSeq.empty, FastSeq(Cast(x, TInt64)), sumSig)) ->
        Array(st),
      streamAggScanIR(st)(x => ApplyScanOp(FastSeq.empty, FastSeq(Cast(x, TInt64)), sumSig)) ->
        Array(st),
      RunAgg(
        Begin(FastSeq(
          InitOp(0, FastSeq(Begin(FastSeq(InitOp(0, FastSeq(), pSumSig)))), groupSignature),
          SeqOp(0, FastSeq(I32(1), SeqOp(0, FastSeq(I64(1)), pSumSig)), groupSignature),
        )),
        AggStateValue(0, groupSignature.state),
        FastSeq(groupSignature.state),
      ), {
        val foo = Ref(freshName(), TInt32)
        RunAggScan(
          StreamRange(I32(0), I32(1), I32(1)),
          foo.name,
          InitOp(0, FastSeq(Begin(FastSeq(InitOp(0, FastSeq(), pSumSig)))), groupSignature),
          SeqOp(0, FastSeq(foo, SeqOp(0, FastSeq(I64(1)), pSumSig)), groupSignature),
          AggStateValue(0, groupSignature.state),
          FastSeq(groupSignature.state),
        )
      },
      AggFilter(True(), I32(0), false) -> (_.createAgg),
      aggExplodeIR(NA(TStream(TInt32)))(_ => I32(0)) -> (_.createAgg),
      AggGroupBy(True(), I32(0), false) -> (_.createAgg),
      ApplyAggOp(FastSeq.empty, FastSeq(I32(0)), collectSig) -> (_.createAgg),
      ApplyAggOp(FastSeq(I32(2)), FastSeq(call), callStatsSig) -> (_.createAgg.bindAgg(
        call.name,
        call.typ,
      )),
      ApplyAggOp(FastSeq(I32(10)), FastSeq(F64(-2.11), I32(4)), takeBySig) -> (_.createAgg),
      AggFold(I32(0), l + I32(1), l + r, l.name, r.name, false) -> (_.createAgg),
      InitOp(0, FastSeq(I32(2)), pCallStatsSig),
      SeqOp(0, FastSeq(i), pCollectSig),
      CombOp(0, 1, pCollectSig),
      ResultOp(0, pCollectSig), {
        val accum = Ref(freshName(), TInt32)
        ResultOp(
          0,
          PhysicalAggSig(
            Fold(),
            FoldStateSig(EmitType(SInt32, true), accum.name, freshName(), accum),
          ),
        )
      },
      SerializeAggs(0, 0, BufferSpec.default, FastSeq(pCollectSig.state)),
      DeserializeAggs(0, 0, BufferSpec.default, FastSeq(pCollectSig.state)),
      CombOpValue(0, bin, pCollectSig) -> Array(bin),
      AggStateValue(0, pCollectSig.state),
      InitFromSerializedValue(0, bin, pCollectSig.state) -> Array(bin),
      Begin(FastSeq(Void())),
      MakeStruct(FastSeq("x" -> i)),
      SelectFields(s, FastSeq("x", "z")) -> Array(s),
      InsertFields(s, FastSeq("x" -> i)) -> Array(s),
      InsertFields(s, FastSeq("* x *" -> i)) -> Array(s), // Won't parse as a simple identifier
      GetField(s, "x") -> Array(s),
      MakeTuple(FastSeq(2 -> i, 4 -> b)),
      GetTupleElement(t, 1) -> Array(t),
      Die("mumblefoo", TFloat64),
      Trap(Die("mumblefoo", TFloat64)),
      invoke("land", TBoolean, b, c) -> Array(c), // ApplySpecial
      invoke("toFloat64", TFloat64, i), // Apply
      Literal(TStruct("x" -> TInt32), Row(1)),
      TableCount(table),
      MatrixCount(mt),
      TableGetGlobals(table),
      TableCollect(TableKeyBy(table, FastSeq())),
      TableAggregate(table, MakeStruct(IndexedSeq("foo" -> count))),
      TableToValueApply(table, ForceCountTable()),
      MatrixToValueApply(mt, ForceCountMatrixTable()),
      TableWrite(table, TableNativeWriter("/path/to/data.ht")),
      MatrixWrite(mt, MatrixNativeWriter("/path/to/data.mt")),
      MatrixWrite(vcf, MatrixVCFWriter("/path/to/sample.vcf")),
      MatrixWrite(vcf, MatrixPLINKWriter("/path/to/base")),
      MatrixWrite(bgen, MatrixGENWriter("/path/to/base")),
      MatrixWrite(mt, MatrixBlockMatrixWriter("path/to/data/bm", true, "a", 4096)),
      MatrixMultiWrite(
        Array(mt, mt),
        MatrixNativeMultiWriter(IndexedSeq("/path/to/mt1", "/path/to/mt2")),
      ),
      TableMultiWrite(
        Array(table, table),
        WrappedMatrixNativeMultiWriter(
          MatrixNativeMultiWriter(IndexedSeq("/path/to/mt1", "/path/to/mt2")),
          FastSeq("foo"),
        ),
      ),
      MatrixAggregate(mt, MakeStruct(IndexedSeq("foo" -> count))),
      BlockMatrixCollect(blockMatrix),
      BlockMatrixWrite(blockMatrix, blockMatrixWriter),
      BlockMatrixMultiWrite(IndexedSeq(blockMatrix, blockMatrix), blockMatrixMultiWriter),
      BlockMatrixWrite(blockMatrix, BlockMatrixPersistWriter("x", "MEMORY_ONLY")),
      cdaIR(rangeIR(3), 1, "test", NA(TString))((context, global) => context),
      ReadPartition(
        MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> Str("foo"))),
        TStruct("foo" -> TInt32),
        PartitionNativeReader(
          TypedCodecSpec(
            ctx,
            PCanonicalStruct("foo" -> PInt32(), "bar" -> PCanonicalString()),
            BufferSpec.default,
          ),
          "rowUID",
        ),
      ),
      WritePartition(
        MakeStream(FastSeq(), TStream(TStruct())),
        NA(TString),
        PartitionNativeWriter(
          TypedCodecSpec(ctx, PType.canonical(TStruct()), BufferSpec.default),
          IndexedSeq(),
          "path",
          None,
          None,
        ),
      ),
      WriteMetadata(
        Begin(FastSeq()),
        RelationalWriter("path", overwrite = false, None),
      ),
      ReadValue(
        Str("foo"),
        ETypeValueReader(TypedCodecSpec(
          ctx,
          PCanonicalStruct("foo" -> PInt32(), "bar" -> PCanonicalString()),
          BufferSpec.default,
        )),
        TStruct("foo" -> TInt32),
      ),
      WriteValue(
        I32(1),
        Str("foo"),
        ETypeValueWriter(TypedCodecSpec(ctx, PInt32(), BufferSpec.default)),
      ),
      WriteValue(
        I32(1),
        Str("foo"),
        ETypeValueWriter(TypedCodecSpec(ctx, PInt32(), BufferSpec.default)),
        Some(Str("/tmp/uid/part")),
      ),
      LiftMeOut(I32(1)),
      relationalBindIR(I32(0))(_ => I32(0)), {
        val y = freshName()
        TailLoop(y, IndexedSeq(freshName() -> I32(0)), TInt32, Recur(y, FastSeq(I32(4)), TInt32))
      },
    )
    val emptyEnv = BindingEnv.empty[Type]
    irs.map { case (ir, bind) => Array(ir, bind(emptyEnv)) }
  }

  @DataProvider(name = "tableIRs")
  def tableIRs(): Array[Array[TableIR]] =
    tableIRs(ctx)

  def tableIRs(ctx: ExecuteContext): Array[Array[TableIR]] = {
    try {
      val fs = ctx.fs

      val read = TableIR.read(fs, getTestResource("backward_compatability/1.1.0/table/0.ht"))
      val mtRead =
        MatrixIR.read(fs, getTestResource("backward_compatability/1.0.0/matrix_table/0.hmt"))
      val b = True()

      val xs: Array[TableIR] = Array(
        TableDistinct(read),
        TableKeyBy(read, Array("m", "d")),
        TableFilter(read, b),
        read,
        MatrixColsTable(mtRead),
        TableAggregateByKey(
          read,
          MakeStruct(FastSeq(
            "a" -> I32(5)
          )),
        ),
        TableKeyByAndAggregate(read, NA(TStruct.empty), NA(TStruct.empty), Some(1), 2),
        TableJoin(read, TableRange(100, 10), "inner", 1),
        TableLeftJoinRightDistinct(read, TableRange(100, 10), "root"),
        TableMultiWayZipJoin(FastSeq(read, read), " * data * ", "globals"),
        MatrixEntriesTable(mtRead),
        MatrixRowsTable(mtRead),
        TableRepartition(read, 10, RepartitionStrategy.COALESCE),
        TableHead(read, 10),
        TableTail(read, 10),
        TableParallelize(
          MakeStruct(FastSeq(
            "rows" -> MakeArray(
              FastSeq(
                MakeStruct(FastSeq("a" -> NA(TInt32))),
                MakeStruct(FastSeq("a" -> I32(1))),
              ),
              TArray(TStruct("a" -> TInt32)),
            ),
            "global" -> MakeStruct(FastSeq()),
          )),
          None,
        ),
        TableMapRows(
          TableKeyBy(read, FastSeq()),
          MakeStruct(FastSeq(
            "a" -> GetField(Ref(TableIR.rowName, read.typ.rowType), "f32"),
            "b" -> F64(-2.11),
          )),
        ), {
          val rs = Ref(freshName(), TStream(read.typ.rowType))
          TableMapPartitions(
            TableKeyBy(read, FastSeq()),
            freshName(),
            rs.name,
            StreamTake(rs, 1),
            0,
            0,
          )
        },
        TableMapGlobals(
          read,
          MakeStruct(FastSeq(
            "foo" -> NA(TArray(TInt32))
          )),
        ),
        TableRange(100, 10),
        TableUnion(
          FastSeq(TableRange(100, 10), TableRange(50, 10))
        ),
        TableExplode(read, Array("mset")),
        TableOrderBy(
          TableKeyBy(read, FastSeq()),
          FastSeq(SortField("m", Ascending), SortField("m", Descending)),
        ),
        CastMatrixToTable(mtRead, " # entries", " # cols"),
        TableRename(read, Map("idx" -> "idx_foo"), Map("global_f32" -> "global_foo")),
        TableFilterIntervals(
          read,
          FastSeq(Interval(IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(10), 1))),
          keep = false,
        ),
        RelationalLetTable(freshName(), I32(0), read), {
          val structs = MakeStream(FastSeq(), TStream(TStruct()))
          val partitioner = RVDPartitioner.empty(ctx.stateManager, TStruct())
          TableGen(
            structs,
            MakeStruct(FastSeq()),
            freshName(),
            freshName(),
            structs,
            partitioner,
            errorId = 180,
          )
        },
      )
      xs.map(x => Array(x))
    } catch {
      case t: Throwable =>
        println(t)
        println(t.printStackTrace())
        throw t
    }
  }

  @DataProvider(name = "matrixIRs")
  def matrixIRs(): Array[Array[MatrixIR]] =
    matrixIRs(ctx)

  def matrixIRs(ctx: ExecuteContext): Array[Array[MatrixIR]] = {
    try {
      val fs = ctx.fs

      CompileAndEvaluate(
        ctx,
        invoke(
          "index_bgen",
          TInt64,
          Array[Type](TLocus("GRCh37")),
          Str(getTestResource("example.8bits.bgen")),
          Str(getTestResource("example.8bits.bgen.idx2")),
          Literal(TDict(TString, TString), Map("01" -> "1")),
          False(),
          I32(1000000),
        ),
      )

      val tableRead = TableIR.read(fs, getTestResource("backward_compatability/1.1.0/table/0.ht"))
      val read =
        MatrixIR.read(fs, getTestResource("backward_compatability/1.0.0/matrix_table/0.hmt"))
      val range = MatrixIR.range(3, 7, None)
      val vcf = importVCF(ctx, getTestResource("sample.vcf"))

      val bgenReader = MatrixBGENReader(
        ctx,
        FastSeq(getTestResource("example.8bits.bgen")),
        None,
        Map.empty[String, String],
        None,
        None,
        None,
      )
      val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

      val range1 = MatrixIR.range(20, 2, Some(3))
      val range2 = MatrixIR.range(20, 2, Some(4))

      val b = True()

      val newCol = MakeStruct(FastSeq(
        "col_idx" -> GetField(Ref(MatrixIR.colName, read.typ.colType), "col_idx"),
        "new_f32" -> ApplyBinaryPrimOp(
          Add(),
          GetField(Ref(MatrixIR.colName, read.typ.colType), "col_f32"),
          F32(-5.2f),
        ),
      ))
      val newRow = MakeStruct(FastSeq(
        "row_idx" -> GetField(Ref(MatrixIR.rowName, read.typ.rowType), "row_idx"),
        "new_f32" -> ApplyBinaryPrimOp(
          Add(),
          GetField(Ref(MatrixIR.rowName, read.typ.rowType), "row_f32"),
          F32(-5.2f),
        ),
      ))

      val collectSig = AggSignature(Collect(), IndexedSeq(), IndexedSeq(TInt32))
      val collect = ApplyAggOp(FastSeq.empty, FastSeq(I32(0)), collectSig)

      val newRowAnn = MakeStruct(FastSeq("count_row" -> collect))
      val newColAnn = MakeStruct(FastSeq("count_col" -> collect))
      val newEntryAnn = MakeStruct(FastSeq("count_entry" -> collect))

      val xs = Array[MatrixIR](
        read,
        MatrixFilterRows(read, b),
        MatrixFilterCols(read, b),
        MatrixFilterEntries(read, b),
        MatrixChooseCols(read, Array(0, 0, 0)),
        MatrixMapCols(read, newCol, None),
        MatrixKeyRowsBy(read, FastSeq("row_m", "row_d"), false),
        MatrixMapRows(read, newRow),
        MatrixRepartition(read, 10, 0),
        MatrixMapEntries(
          read,
          MakeStruct(FastSeq(
            "global_f32" -> ApplyBinaryPrimOp(
              Add(),
              GetField(Ref(MatrixIR.globalName, read.typ.globalType), "global_f32"),
              F32(-5.2f),
            )
          )),
        ),
        MatrixCollectColsByKey(read),
        MatrixAggregateColsByKey(read, newEntryAnn, newColAnn),
        MatrixAggregateRowsByKey(read, newEntryAnn, newRowAnn),
        range,
        vcf,
        bgen,
        MatrixExplodeRows(read, FastSeq("row_mset")),
        MatrixUnionRows(FastSeq(range1, range2)),
        MatrixDistinctByRow(range1),
        MatrixRowsHead(range1, 3),
        MatrixColsHead(range1, 3),
        MatrixRowsTail(range1, 3),
        MatrixColsTail(range1, 3),
        MatrixExplodeCols(read, FastSeq("col_mset")),
        CastTableToMatrix(
          CastMatrixToTable(read, " # entries", " # cols"),
          " # entries",
          " # cols",
          read.typ.colKey,
        ),
        MatrixAnnotateColsTable(read, tableRead, "uid_123"),
        MatrixAnnotateRowsTable(read, tableRead, "uid_123", product = false),
        MatrixRename(
          read,
          Map("global_i64" -> "foo"),
          Map("col_i64" -> "bar"),
          Map("row_i64" -> "baz"),
          Map("entry_i64" -> "quam"),
        ),
        MatrixFilterIntervals(
          read,
          FastSeq(Interval(IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(10), 1))),
          keep = false,
        ),
        RelationalLetMatrixTable(freshName(), I32(0), read),
      )

      xs.map(x => Array(x))
    } catch {
      case t: Throwable =>
        println(t)
        println(t.printStackTrace())
        throw t
    }
  }

  @DataProvider(name = "blockMatrixIRs")
  def blockMatrixIRs(): Array[Array[BlockMatrixIR]] = {
    val read =
      BlockMatrixRead(BlockMatrixNativeReader(fs, getTestResource("blockmatrix_example/0")))
    val transpose = BlockMatrixBroadcast(read, FastSeq(1, 0), FastSeq(2, 2), 2)
    val dot = BlockMatrixDot(read, transpose)
    val slice = BlockMatrixSlice(read, FastSeq(FastSeq(0, 2, 1), FastSeq(0, 1, 1)))

    val sparsify1 = BlockMatrixSparsify(read, RectangleSparsifier(FastSeq(FastSeq(0L, 1L, 5L, 6L))))
    val sparsify2 = BlockMatrixSparsify(read, BandSparsifier(true, -1L, 1L))
    val sparsify3 = BlockMatrixSparsify(
      read,
      RowIntervalSparsifier(true, FastSeq(0L, 1L, 5L, 6L), FastSeq(5L, 6L, 8L, 9L)),
    )
    val densify = BlockMatrixDensify(read)

    val blockMatrixIRs = Array[BlockMatrixIR](
      read,
      transpose,
      dot,
      sparsify1,
      sparsify2,
      sparsify3,
      densify,
      RelationalLetBlockMatrix(freshName(), I32(0), read),
      slice,
    )

    blockMatrixIRs.map(ir => Array(ir))
  }

  @Test def testIRConstruction(): scalatest.Assertion = {
    matrixIRs()
    tableIRs()
    valueIRs()
    blockMatrixIRs()
    scalatest.Succeeded
  }

  @Test(dataProvider = "valueIRs")
  def testValueIRParser(x: IR, refMap: BindingEnv[Type]): scalatest.Assertion = {
    val s = Pretty.sexprStyle(x, elideLiterals = false)
    val x2 = IRParser.parse_value_ir(ctx, s, refMap)
    assert(x2 == x)
  }

  @Test(dataProvider = "tableIRs")
  def testTableIRParser(x: TableIR): scalatest.Assertion = {
    val s = Pretty.sexprStyle(x, elideLiterals = false)
    val x2 = IRParser.parse_table_ir(ctx, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "matrixIRs")
  def testMatrixIRParser(x: MatrixIR): scalatest.Assertion = {
    val s = Pretty.sexprStyle(x, elideLiterals = false)
    val x2 = IRParser.parse_matrix_ir(ctx, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "blockMatrixIRs")
  def testBlockMatrixIRParser(x: BlockMatrixIR): scalatest.Assertion = {
    val s = Pretty.sexprStyle(x, elideLiterals = false)
    val x2 = IRParser.parse_blockmatrix_ir(ctx, s)
    assert(x2 == x)
  }

  @Test def testBlockMatrixIRParserPersist(): scalatest.Assertion = {
    val cache = mutable.Map.empty[String, BlockMatrix]
    val bm = BlockMatrixRandom(0, gaussian = true, shape = Array(5L, 6L), blockSize = 3)
    try {
      ctx.local(blockMatrixCache = cache) { ctx =>
        backend.execute(ctx, BlockMatrixWrite(bm, BlockMatrixPersistWriter("x", "MEMORY_ONLY")))
      }

      ctx.local(blockMatrixCache = cache) { ctx =>
        val persist = BlockMatrixRead(BlockMatrixPersistReader("x", bm.typ))
        val s = Pretty.sexprStyle(persist, elideLiterals = false)
        val x2 = IRParser.parse_blockmatrix_ir(ctx, s)
        assert(x2 == persist)
      }
    } finally
      cache.values.foreach(_.unpersist())
  }

  @Test def testCachedIR(): scalatest.Assertion = {
    val cached = Literal(TSet(TInt32), Set(1))
    val s = s"(JavaIR 1)"
    val x2 =
      ctx.local(persistedIrCache = mutable.Map(1 -> cached))(ctx => IRParser.parse_value_ir(ctx, s))
    assert(x2 eq cached)
  }

  @Test def testCachedTableIR(): scalatest.Assertion = {
    val cached = TableRange(1, 1)
    val s = s"(JavaTable 1)"
    val x2 =
      ctx.local(persistedIrCache = mutable.Map(1 -> cached))(ctx => IRParser.parse_table_ir(ctx, s))
    assert(x2 eq cached)
  }

  @Test def testArrayContinuationDealsWithIfCorrectly(): scalatest.Assertion = {
    val ir = ToArray(mapIR(
      If(IsNA(In(0, TBoolean)), NA(TStream(TInt32)), ToStream(In(1, TArray(TInt32))))
    )(Cast(_, TInt64)))

    assertEvalsTo(ir, FastSeq(true -> TBoolean, FastSeq(0) -> TArray(TInt32)), FastSeq(0L))
  }

  @Test def testTableGetGlobalsSimplifyRules(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t1 =
      TableType(TStruct("a" -> TInt32), FastSeq("a"), TStruct("g1" -> TInt32, "g2" -> TFloat64))
    val t2 =
      TableType(TStruct("a" -> TInt32), FastSeq("a"), TStruct("g3" -> TInt32, "g4" -> TFloat64))
    val tab1 = TableLiteral(
      TableValue(
        ctx,
        t1,
        BroadcastRow(ctx, Row(1, 1.1), t1.globalType),
        RVD.empty(ctx, t1.canonicalRVDType),
      ),
      theHailClassLoader,
    )
    val tab2 = TableLiteral(
      TableValue(
        ctx,
        t2,
        BroadcastRow(ctx, Row(2, 2.2), t2.globalType),
        RVD.empty(ctx, t2.canonicalRVDType),
      ),
      theHailClassLoader,
    )

    assertEvalsTo(TableGetGlobals(TableJoin(tab1, tab2, "left")), Row(1, 1.1, 2, 2.2))
    assertEvalsTo(
      TableGetGlobals(TableMapGlobals(
        tab1,
        InsertFields(Ref(TableIR.globalName, t1.globalType), IndexedSeq("g1" -> I32(3))),
      )),
      Row(3, 1.1),
    )
    assertEvalsTo(TableGetGlobals(TableRename(tab1, Map.empty, Map("g2" -> "g3"))), Row(1, 1.1))
  }

  @Test def testAggLet(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly
    val t = TableRange(2, 2)
    val ir = TableAggregate(
      t,
      aggBindIR(GetField(Ref(TableIR.rowName, t.typ.rowType), "idx").toL + I64(1)) { a =>
        aggBindIR(a * I64(2))(b => ApplyAggOp(Max())(b * b)) + aggBindIR(a * I64(3)) { c =>
          ApplyAggOp(Sum())(c * c)
        }
      },
    )

    assertEvalsTo(ir, 61L)
  }

  @Test def testRelationalLet(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly

    val ir = relationalBindIR(NA(TInt32))(x => x)
    assertEvalsTo(ir, null)
  }

  @Test def testRelationalLetTable(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32))
    val x = RelationalRef(freshName(), t)
    val ir = TableAggregate(
      RelationalLetTable(
        x.name,
        Literal(t, FastSeq(Row(1))),
        TableParallelize(MakeStruct(FastSeq(
          "rows" -> x,
          "global" -> MakeStruct(FastSeq()),
        ))),
      ),
      ApplyAggOp(FastSeq(), FastSeq(), AggSignature(Count(), FastSeq(), FastSeq())),
    )
    assertEvalsTo(ir, 1L)
  }

  @Test def testRelationalLetMatrixTable(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32))
    val x = RelationalRef(freshName(), t)
    val m = CastTableToMatrix(
      TableMapGlobals(
        TableMapRows(
          TableRange(1, 1),
          InsertFields(
            Ref(TableIR.rowName, TStruct("idx" -> TInt32)),
            FastSeq("entries" -> x),
          ),
        ),
        MakeStruct(FastSeq("cols" -> MakeArray(
          FastSeq(MakeStruct(FastSeq("s" -> I32(0)))),
          TArray(TStruct("s" -> TInt32)),
        ))),
      ),
      "entries",
      "cols",
      FastSeq(),
    )
    val ir = MatrixAggregate(
      RelationalLetMatrixTable(x.name, Literal(t, FastSeq(Row(1))), m),
      ApplyAggOp(FastSeq(), FastSeq(), AggSignature(Count(), FastSeq(), FastSeq())),
    )
    assertEvalsTo(ir, 1L)
  }

  @DataProvider(name = "relationalFunctions")
  def relationalFunctionsData(): Array[Array[Any]] = Array(
    Array(TableFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(VEP(fs, getTestResource("dummy_vep_config.json"), false, 1, true)),
    Array(WrappedMatrixToTableFunction(
      LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b")),
      "foo",
      "bar",
      FastSeq("ck"),
    )),
    Array(LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b"))),
    Array(LinearRegressionRowsChained(
      FastSeq(FastSeq("foo")),
      "bar",
      Array("baz"),
      1,
      Array("a", "b"),
    )),
    Array(LogisticRegression(
      "firth",
      Array("a", "b"),
      "c",
      Array("d", "e"),
      Array("f", "g"),
      25,
      1e-6,
    )),
    Array(PoissonRegression("firth", "a", "c", Array("d", "e"), Array("f", "g"), 25, 1e-6)),
    Array(Skat("a", "b", "c", "d", Array("e", "f"), false, 1, 0.1, 100, 0, 0.0)),
    Array(LocalLDPrune("x", 0.95, 123, 456)),
    Array(PCA("x", 1, false)),
    Array(PCRelate(0.00, 4096, Some(0.1), PCRelate.PhiK2K0K1)),
    Array(MatrixFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(ForceCountTable()),
    Array(ForceCountMatrixTable()),
    Array(NPartitionsTable()),
    Array(NPartitionsMatrixTable()),
    Array(WrappedMatrixToValueFunction(NPartitionsMatrixTable(), "foo", "bar", FastSeq("a", "c"))),
    Array(MatrixExportEntriesByCol(1, "asd", false, true, false)),
    Array(GetElement(FastSeq(1, 2))),
  )

  @Test def relationalFunctionsRun(): scalatest.Assertion = {
    relationalFunctionsData(); scalatest.Succeeded
  }

  @Test(dataProvider = "relationalFunctions")
  def testRelationalFunctionsSerialize(x: Any): scalatest.Assertion = {
    implicit val formats = RelationalFunctions.formats

    x match {
      case x: MatrixToMatrixFunction =>
        assert(RelationalFunctions.lookupMatrixToMatrix(ctx, Serialization.write(x)) == x)
      case x: MatrixToTableFunction =>
        assert(RelationalFunctions.lookupMatrixToTable(ctx, Serialization.write(x)) == x)
      case x: MatrixToValueFunction =>
        assert(RelationalFunctions.lookupMatrixToValue(ctx, Serialization.write(x)) == x)
      case x: TableToTableFunction =>
        assert(RelationalFunctions.lookupTableToTable(ctx, JsonMethods.compact(x.toJValue)) == x)
      case x: TableToValueFunction =>
        assert(RelationalFunctions.lookupTableToValue(ctx, Serialization.write(x)) == x)
      case x: BlockMatrixToTableFunction =>
        assert(RelationalFunctions.lookupBlockMatrixToTable(ctx, Serialization.write(x)) == x)
      case x: BlockMatrixToValueFunction =>
        assert(RelationalFunctions.lookupBlockMatrixToValue(ctx, Serialization.write(x)) == x)
    }
  }

  @Test def testFoldWithSetup(): scalatest.Assertion = {
    val v = In(0, TInt32)
    val cond1 = If(
      v.ceq(I32(3)),
      MakeStream(FastSeq(I32(1), I32(2), I32(3)), TStream(TInt32)),
      MakeStream(FastSeq(I32(4), I32(5), I32(6)), TStream(TInt32)),
    )
    assertEvalsTo(
      foldIR(cond1, True())((_, i) => i ceq v),
      FastSeq(0 -> TInt32),
      false,
    )
  }

  @Test def testNonCanonicalTypeParsing(): scalatest.Assertion = {
    val t = TTuple(FastSeq(TupleField(1, TInt64)))
    val lit = Literal(t, Row(1L))

    assert(IRParser.parseType(t.parsableString()) == t)
    assert(IRParser.parse_value_ir(ctx, Pretty.sexprStyle(lit, elideLiterals = false)) == lit)
  }

  def regressionTestUnifyBug(): scalatest.Assertion = {
    // failed due to misuse of Type.unify
    val ir = IRParser.parse_value_ir(
      ctx,
      """
        |(ToArray (StreamMap __uid_3
        |    (ToStream (Literal Array[Interval[Locus(GRCh37)]] "[{\"start\": {\"contig\": \"20\", \"position\": 10277621}, \"end\": {\"contig\": \"20\", \"position\": 11898992}, \"includeStart\": true, \"includeEnd\": false}]"))
        |    (Apply Interval Interval[Struct{locus:Locus(GRCh37)}]
        |       (MakeStruct (locus  (Apply start Locus(GRCh37) (Ref __uid_3))))
        |       (MakeStruct (locus  (Apply end Locus(GRCh37) (Ref __uid_3)))) (True) (False))))
        |""".stripMargin,
    )
    val v = backend.execute(ctx, ir)

    assert(
      ir.typ.ordering(ctx.stateManager).equiv(
        FastSeq(
          Interval(
            Row(Locus("20", 10277621)),
            Row(Locus("20", 11898992)),
            includesStart = true,
            includesEnd = false,
          )
        ),
        v,
      )
    )
  }

  @Test def testSimpleTailLoop(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly
    val triangleSum: IR =
      tailLoop(TInt32, In(0, TInt32), In(1, TInt32)) { case (recur, Seq(x, accum)) =>
        If(x <= 0, accum, recur(FastSeq(x - 1, accum + x)))
      }

    assertEvalsTo(triangleSum, FastSeq(5 -> TInt32, 0 -> TInt32), 15)
    assertEvalsTo(triangleSum, FastSeq(5 -> TInt32, (null, TInt32)), null)
    assertEvalsTo(triangleSum, FastSeq((null, TInt32), 0 -> TInt32), null)
  }

  @Test def testNestedTailLoop(): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly
    val triangleSum: IR = tailLoop(TInt32, In(0, TInt32), I32(0)) { case (recur, Seq(x, accum)) =>
      If(
        x <= 0,
        tailLoop(TInt32, accum, I32(0)) { case (recur2, Seq(x2, accum2)) =>
          If(x2 <= 0, accum2, recur2(FastSeq(x2 - 5, accum2 + x2)))
        },
        recur(FastSeq(x - I32(1), accum + x)),
      )
    }

    assertEvalsTo(triangleSum, FastSeq(5 -> TInt32), 15 + 10 + 5)
  }

  @Test def testTailLoopNDMemory(): scalatest.Assertion = {
    val ndType = TNDArray(TInt32, Nat(2))

    val ndSum: IR = tailLoop(
      ndType,
      In(0, TInt32),
      In(1, ndType),
    ) { case (recur, Seq(x, accum)) =>
      If(
        x <= 0,
        accum,
        recur(FastSeq(x - 1, ndMap(accum)(elt => elt + x))),
      )
    }

    val startingArg = SafeNDArray(IndexedSeq[Long](4L, 4L), (0 until 16).toFastSeq)

    val memUsed = RegionPool.scoped { pool =>
      ctx.local(r = Region(pool = pool)) { ctx =>
        eval(ndSum, Env.empty, FastSeq(2 -> TInt32, startingArg -> ndType), None, None, true, ctx)
        pool.getHighestTotalUsage
      }
    }

    RegionPool.scoped { pool =>
      ctx.local(r = Region(pool = pool)) { ctx =>
        eval(ndSum, Env.empty, FastSeq(100 -> TInt32, startingArg -> ndType), None, None, true, ctx)
        assert(memUsed == pool.getHighestTotalUsage)
      }
    }
  }

  @Test def testHasIRSharing(): scalatest.Assertion = {
    val r = Ref(freshName(), TInt32)
    val ir1 = MakeTuple.ordered(FastSeq(I64(1), r, r, I32(1)))
    assert(HasIRSharing(ctx)(ir1))
    assert(!HasIRSharing(ctx)(ir1.deepCopy()))
  }

  @Test def freeVariables(): scalatest.Assertion = {
    val stream = rangeIR(5)
    val sumSig = AggSignature(Sum(), IndexedSeq(), IndexedSeq(TInt32))
    val y = Ref(freshName(), TInt32)
    val z = Ref(freshName(), TInt32)
    val explodeIR = aggExplodeIR(stream) { x =>
      z + ApplyAggOp(FastSeq.empty, FastSeq(x + y), sumSig)
    }

    assert(FreeVariables(explodeIR, true, true) == BindingEnv[Unit](
      Env((z.name, ())),
      Some(Env((y.name, ()))),
      Some(Env()),
    ))
    assert(FreeVariables(explodeIR, false, false) == BindingEnv[Unit](Env((z.name, ()))))

    val aggIR = streamAggIR(stream)(x => z + ApplyAggOp(FastSeq.empty, FastSeq(x + y), sumSig))
    assert(
      FreeVariables(aggIR, true, true) == BindingEnv[Unit](
        Env((z.name, ()), (y.name, ())),
        Some(Env()),
        Some(Env()),
      )
    )
  }

  @Test def freeVariablesAggScanBindingEnv(): scalatest.Assertion = {
    def testFreeVarsHelper(ir: IR): scalatest.Assertion = {
      val irFreeVarsTrue = FreeVariables.apply(ir, true, true)
      assert(irFreeVarsTrue.agg.isDefined && irFreeVarsTrue.scan.isDefined)

      val irFreeVarsFalse = FreeVariables.apply(ir, false, false)
      assert(irFreeVarsFalse.agg.isEmpty && irFreeVarsFalse.scan.isEmpty)
    }

    val aggIR = streamAggIR(mapIR(rangeIR(4))(Cast(_, TInt64)))(ApplyAggOp(Sum())(_))
    testFreeVarsHelper(aggIR)

    val scanIR = streamAggScanIR(Ref(freshName(), TStream(TInt32))) { x =>
      ApplyScanOp(Sum())(Cast(x, TInt64))
    }
    testFreeVarsHelper(scanIR)
  }

  @DataProvider(name = "nonNullTypesAndValues")
  def nonNullTypesAndValues(): Array[Array[Any]] = Array(
    Array(Int32SingleCodeType, 1),
    Array(Int64SingleCodeType, 5L),
    Array(Float32SingleCodeType, 5.5f),
    Array(Float64SingleCodeType, 1.2),
    Array(PTypeReferenceSingleCodeType(PCanonicalString()), "foo"),
    Array(PTypeReferenceSingleCodeType(PCanonicalArray(PInt32())), FastSeq(5, 7, null, 3)),
    Array(
      PTypeReferenceSingleCodeType(PCanonicalTuple(
        false,
        PInt32(),
        PCanonicalString(),
        PCanonicalStruct(),
      )),
      Row(3, "bar", Row()),
    ),
  )

  @Test(dataProvider = "nonNullTypesAndValues")
  def testReadWriteValues(pt: SingleCodeType, value: Any): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly
    val node = In(0, SingleCodeEmitParamType(true, pt))
    val spec = TypedCodecSpec(ctx, PType.canonical(node.typ), BufferSpec.blockedUncompressed)
    val writer = ETypeValueWriter(spec)
    val reader = ETypeValueReader(spec)
    val prefix = ctx.createTmpPath("test-read-write-values")
    val filename = WriteValue(node, Str(prefix) + UUID4(), writer)
    forAll(Array(value, null)) { v =>
      assertEvalsTo(ReadValue(filename, reader, pt.virtualType), FastSeq(v -> pt.virtualType), v)
    }
  }

  @Test(dataProvider = "nonNullTypesAndValues")
  def testReadWriteValueDistributed(pt: SingleCodeType, value: Any): scalatest.Assertion = {
    implicit val execStrats = ExecStrategy.compileOnly
    val node = In(0, SingleCodeEmitParamType(true, pt))
    val spec = TypedCodecSpec(ctx, PType.canonical(node.typ), BufferSpec.blockedUncompressed)
    val writer = ETypeValueWriter(spec)
    val reader = ETypeValueReader(spec)
    val prefix = ctx.createTmpPath("test-read-write-value-dist")
    val readArray = bindIR(
      cdaIR(
        mapIR(rangeIR(10))(_ => node),
        MakeStruct(FastSeq()),
        "test",
        NA(TString),
      )((ctx, _) => WriteValue(ctx, Str(prefix) + UUID4(), writer))
    )(files => mapIR(ToStream(files))(ReadValue(_, reader, pt.virtualType)))

    forAll(Array(value, null)) { v =>
      assertEvalsTo(ToArray(readArray), FastSeq(v -> pt.virtualType), Array.fill(10)(v).toFastSeq)
    }
  }

  @Test def testUUID4(): scalatest.Assertion = {
    val single = UUID4()
    val hex = "[0-9a-f]"
    val format = s"$hex{8}-$hex{4}-$hex{4}-$hex{4}-$hex{12}"
    // 12345678-1234-5678-1234-567812345678
    assertEvalsTo(
      bindIR(single) { s =>
        invoke("regexMatch", TBoolean, Str(format), s) &&
        invoke("length", TInt32, s).ceq(I32(36))
      },
      true,
    )

    val stream = mapIR(rangeIR(5))(_ => single)

    def selfZip(s: IR, n: Int) = zipIR(
      Array.fill(n)(s),
      ArrayZipBehavior.AssumeSameLength,
    )(MakeArray(_, TArray(TString)))

    def assertNumDistinct(s: IR, expected: Int) =
      assertEvalsTo(ArrayLen(CastToArray(ToSet(s))), expected)

    assertNumDistinct(stream, 5)
    assertNumDistinct(flatten(selfZip(stream, 2)), 10)
    assertNumDistinct(bindIR(ToArray(stream))(a => selfZip(ToStream(a), 2)), 5)
  }

  @Test def testZipDoesntPruneLengthInfo(): scalatest.Assertion =
    forAll {
      Array(
        ArrayZipBehavior.AssumeSameLength,
        ArrayZipBehavior.AssertSameLength,
        ArrayZipBehavior.TakeMinLength,
        ArrayZipBehavior.ExtendNA,
      )
    } { behavior =>
      val zip = zipIR(
        FastSeq(StreamRange(0, 10, 1), StreamRange(0, 10, 1)),
        behavior,
      )(_ => makestruct("x" -> Str("foo"), "y" -> Str("bar")))

      assertEvalsTo(ToArray(zip), Array.fill(10)(Row("foo", "bar")).toFastSeq)
    }

  @Test def testStreamDistribute(): scalatest.Assertion = {
    val data1 = IndexedSeq(0, 1, 1, 2, 4, 7, 7, 7, 9, 11, 15, 20, 22, 28, 50, 100)
    val pivots1 = IndexedSeq(-10, 1, 7, 7, 15, 22, 50, 200)
    val pivots2 = IndexedSeq(-10, 1, 1, 7, 9, 28, 50, 200)
    val pivots3 = IndexedSeq(-3, 0, 20, 100, 200)
    val pivots4 = IndexedSeq(-8, 4, 7, 7, 150)
    val pivots5 = IndexedSeq(0, 1, 4, 15, 200)
    val pivots6 = IndexedSeq(0, 7, 20, 100)

    runStreamDistTest(data1, pivots1)
    runStreamDistTest(data1, pivots2)
    runStreamDistTest(data1, pivots3)
    runStreamDistTest(data1, pivots4)
    runStreamDistTest(data1, pivots5)
    runStreamDistTest(data1, pivots6)

    val data2 = IndexedSeq(0, 2)
    val pivots11 = IndexedSeq(0, 0, 2)

    runStreamDistTest(data2, pivots11)
  }

  def runStreamDistTest(data: IndexedSeq[Int], splitters: IndexedSeq[Int]): scalatest.Assertion = {
    def makeRowStruct(i: Int) =
      MakeStruct(IndexedSeq(("rowIdx", I32(i)), ("extraInfo", I32(i * i))))
    def makeKeyStruct(i: Int) = MakeStruct(IndexedSeq(("rowIdx", I32(i))))
    val child = ToStream(MakeArray(data.map(makeRowStruct): _*))
    val pivots = MakeArray(splitters.map(makeKeyStruct): _*)
    val spec = TypedCodecSpec(
      ctx,
      PCanonicalStruct(("rowIdx", PInt32Required), ("extraInfo", PInt32Required)),
      BufferSpec.default,
    )
    val dist = StreamDistribute(
      child,
      pivots,
      Str(ctx.localTmpdir),
      Compare(pivots.typ.asInstanceOf[TArray].elementType),
      spec,
    )
    val result = eval(dist).asInstanceOf[IndexedSeq[Row]].map(row =>
      (
        row(0).asInstanceOf[Interval],
        row(1).asInstanceOf[String],
        row(2).asInstanceOf[Int],
        row(3).asInstanceOf[Long],
      )
    )
    val kord: ExtendedOrdering =
      PartitionBoundOrdering(ctx, pivots.typ.asInstanceOf[TArray].elementType)

    var dataIdx = 0

    result.foreach { case (interval, path, elementCount, _) =>
      val reader = PartitionNativeReader(spec, "rowUID")
      val read = ToArray(ReadPartition(
        MakeStruct(Array("partitionIndex" -> I64(0), "partitionPath" -> Str(path))),
        tcoerce[TStruct](spec._vType),
        reader,
      ))
      val rowsFromDisk = eval(read).asInstanceOf[IndexedSeq[Row]]
      assert(rowsFromDisk.size == elementCount)
      assert(rowsFromDisk.forall(interval.contains(kord, _)))

      rowsFromDisk.foreach { row =>
        assert(row(0) == data(dataIdx))
        dataIdx += 1
      }
    }

    assert(dataIdx == data.size)

    result.map(_._1).sliding(2).foreach { case IndexedSeq(interval1, interval2) =>
      assert(interval1.isDisjointFrom(kord, interval2))
    }

    val splitterValueDuplicated = splitters.counter().mapValues(_ > 1)
    val intBuilder = new IntArrayBuilder()
    splitters.toSet.toIndexedSeq.sorted.foreach { e =>
      intBuilder.add(e)
      if (splitterValueDuplicated(e)) {
        intBuilder.add(e)
      }
    }
    val expectedStartsAndEnds = intBuilder.result().sliding(2).toIndexedSeq

    forAll(result.map(_._1).zip(expectedStartsAndEnds)) { case (interval, splitterPair) =>
      assert(interval.start.asInstanceOf[Row](0) == splitterPair(0))
      assert(interval.end.asInstanceOf[Row](0) == splitterPair(1))
    }
  }
}

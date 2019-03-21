package is.hail.expr.ir

import is.hail.ExecStrategy.ExecStrategy
import is.hail.{ExecStrategy, SparkSuite}
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.asm4s.Code
import is.hail.expr.ir
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.ir.IRSuite.TestFunctions
import is.hail.expr.ir.functions.{IRFunctionRegistry, RegistryFunctions, SeededIRFunction}
import is.hail.expr.types.TableType
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.io.bgen.MatrixBGENReader
import is.hail.linalg.BlockMatrix
import is.hail.methods.{ForceCountMatrixTable, ForceCountTable}
import is.hail.rvd.RVD
import is.hail.table.{Ascending, Descending, SortField, Table}
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

import scala.language.{dynamics, implicitConversions}

object IRSuite {
  outer =>
  var globalCounter: Int = 0

  def incr(): Unit = {
    globalCounter += 1
  }

  object TestFunctions extends RegistryFunctions {

    def registerSeededWithMissingness(mname: String, aTypes: Array[Type], rType: Type)(impl: (EmitMethodBuilder, Long, Array[EmitTriplet]) => EmitTriplet) {
      IRFunctionRegistry.addIRFunction(new SeededIRFunction {
        val isDeterministic: Boolean = false

        override val name: String = mname

        override val argTypes: Seq[Type] = aTypes

        override val returnType: Type = rType

        def applySeeded(seed: Long, mb: EmitMethodBuilder, args: EmitTriplet*): EmitTriplet =
          impl(mb, seed, args.toArray)
      })
    }

    def registerSeededWithMissingness(mname: String, mt1: Type, rType: Type)(impl: (EmitMethodBuilder, Long, EmitTriplet) => EmitTriplet): Unit =
      registerSeededWithMissingness(mname, Array(mt1), rType) { case (mb, seed, Array(a1)) => impl(mb, seed, a1) }

    def registerAll() {
      registerSeededWithMissingness("incr_s", TBoolean(), TBoolean()) { (mb, _, l) =>
        EmitTriplet(Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.setup),
          l.m,
          l.v)
      }

      registerSeededWithMissingness("incr_m", TBoolean(), TBoolean()) { (mb, _, l) =>
        EmitTriplet(l.setup,
          Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.m),
          l.v)
      }

      registerSeededWithMissingness("incr_v", TBoolean(), TBoolean()) { (mb, _, l) =>
        EmitTriplet(l.setup,
          l.m,
          Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.v))
      }
    }
  }

}

class IRSuite extends SparkSuite {
  implicit val execStrats = ExecStrategy.values

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
    implicit val execStrats = ExecStrategy.javaOnly

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

  val i32na = NA(TInt32())
  val i64na = NA(TInt64())
  val f32na = NA(TFloat32())
  val f64na = NA(TFloat64())
  val bna = NA(TBoolean())

  @Test def testApplyUnaryPrimOpNegate() {
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), I32(5)), -5)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), i32na), null)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), I64(5)), -5L)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), i64na), null)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), F32(5)), -5F)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), f32na), null)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), F64(5)), -5D)
    assertEvalsTo(ApplyUnaryPrimOp(Negate(), f64na), null)
  }

  @Test def testApplyUnaryPrimOpBang() {
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), False()), true)
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), True()), false)
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), bna), null)
  }

  @Test def testApplyUnaryPrimOpBitFlip() {
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), I32(0xdeadbeef)), ~0xdeadbeef)
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), I32(-0xdeadbeef)), ~(-0xdeadbeef))
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), i32na), null)
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), I64(0xdeadbeef12345678L)), ~0xdeadbeef12345678L)
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), I64(-0xdeadbeef12345678L)), ~(-0xdeadbeef12345678L))
    assertEvalsTo(ApplyUnaryPrimOp(BitNot(), i64na), null)
  }

  @Test def testApplyBinaryPrimOpAdd() {
    def assertSumsTo(t: Type, x: Any, y: Any, sum: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Add(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), sum)
    }
    assertSumsTo(TInt32(), 5, 3, 8)
    assertSumsTo(TInt32(), 5, null, null)
    assertSumsTo(TInt32(), null, 3, null)
    assertSumsTo(TInt32(), null, null, null)

    assertSumsTo(TInt64(), 5L, 3L, 8L)
    assertSumsTo(TInt64(), 5L, null, null)
    assertSumsTo(TInt64(), null, 3L, null)
    assertSumsTo(TInt64(), null, null, null)

    assertSumsTo(TFloat32(), 5.0f, 3.0f, 8.0f)
    assertSumsTo(TFloat32(), 5.0f, null, null)
    assertSumsTo(TFloat32(), null, 3.0f, null)
    assertSumsTo(TFloat32(), null, null, null)

    assertSumsTo(TFloat64(), 5.0, 3.0, 8.0)
    assertSumsTo(TFloat64(), 5.0, null, null)
    assertSumsTo(TFloat64(), null, 3.0, null)
    assertSumsTo(TFloat64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpSubtract() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Subtract(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 3)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 3L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)

    assertExpected(TFloat32(), 5f, 2f, 3f)
    assertExpected(TFloat32(), 5f, null, null)
    assertExpected(TFloat32(), null, 2f, null)
    assertExpected(TFloat32(), null, null, null)

    assertExpected(TFloat64(), 5d, 2d, 3d)
    assertExpected(TFloat64(), 5d, null, null)
    assertExpected(TFloat64(), null, 2d, null)
    assertExpected(TFloat64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpMultiply() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Multiply(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 10)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 10L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)

    assertExpected(TFloat32(), 5f, 2f, 10f)
    assertExpected(TFloat32(), 5f, null, null)
    assertExpected(TFloat32(), null, 2f, null)
    assertExpected(TFloat32(), null, null, null)

    assertExpected(TFloat64(), 5d, 2d, 10d)
    assertExpected(TFloat64(), 5d, null, null)
    assertExpected(TFloat64(), null, 2d, null)
    assertExpected(TFloat64(), null, null, null)

  }

  @Test def testApplyBinaryPrimOpFloatingPointDivide() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(FloatingPointDivide(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 2.5f)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 2.5f)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)

    assertExpected(TFloat32(), 5f, 2f, 2.5f)
    assertExpected(TFloat32(), 5f, null, null)
    assertExpected(TFloat32(), null, 2f, null)
    assertExpected(TFloat32(), null, null, null)

    assertExpected(TFloat64(), 5d, 2d, 2.5d)
    assertExpected(TFloat64(), 5d, null, null)
    assertExpected(TFloat64(), null, 2d, null)
    assertExpected(TFloat64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpRoundToNegInfDivide() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(RoundToNegInfDivide(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 2)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 2L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)

    assertExpected(TFloat32(), 5f, 2f, 2f)
    assertExpected(TFloat32(), 5f, null, null)
    assertExpected(TFloat32(), null, 2f, null)
    assertExpected(TFloat32(), null, null, null)

    assertExpected(TFloat64(), 5d, 2d, 2d)
    assertExpected(TFloat64(), 5d, null, null)
    assertExpected(TFloat64(), null, 2d, null)
    assertExpected(TFloat64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitAnd(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitAnd(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 5 & 2)
    assertExpected(TInt32(), -5, 2, -5 & 2)
    assertExpected(TInt32(), 5, -2, 5 & -2)
    assertExpected(TInt32(), -5, -2, -5 & -2)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 5L & 2L)
    assertExpected(TInt64(), -5L, 2L, -5L & 2L)
    assertExpected(TInt64(), 5L, -2L, 5L & -2L)
    assertExpected(TInt64(), -5L, -2L, -5L & -2L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitOr(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitOr(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 5 | 2)
    assertExpected(TInt32(), -5, 2, -5 | 2)
    assertExpected(TInt32(), 5, -2, 5 | -2)
    assertExpected(TInt32(), -5, -2, -5 | -2)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 5L | 2L)
    assertExpected(TInt64(), -5L, 2L, -5L | 2L)
    assertExpected(TInt64(), 5L, -2L, 5L | -2L)
    assertExpected(TInt64(), -5L, -2L, -5L | -2L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpBitXOr(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitXOr(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32(), 5, 2, 5 ^ 2)
    assertExpected(TInt32(), -5, 2, -5 ^ 2)
    assertExpected(TInt32(), 5, -2, 5 ^ -2)
    assertExpected(TInt32(), -5, -2, -5 ^ -2)
    assertExpected(TInt32(), 5, null, null)
    assertExpected(TInt32(), null, 2, null)
    assertExpected(TInt32(), null, null, null)

    assertExpected(TInt64(), 5L, 2L, 5L ^ 2L)
    assertExpected(TInt64(), -5L, 2L, -5L ^ 2L)
    assertExpected(TInt64(), 5L, -2L, 5L ^ -2L)
    assertExpected(TInt64(), -5L, -2L, -5L ^ -2L)
    assertExpected(TInt64(), 5L, null, null)
    assertExpected(TInt64(), null, 2L, null)
    assertExpected(TInt64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpLeftShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(LeftShift(), In(0, t), In(1, TInt32())), FastIndexedSeq(x -> t, y -> TInt32()), expected)
    }

    assertShiftsTo(TInt32(), 5, 2, 5 << 2)
    assertShiftsTo(TInt32(), -5, 2, -5 << 2)
    assertShiftsTo(TInt32(), 5, null, null)
    assertShiftsTo(TInt32(), null, 2, null)
    assertShiftsTo(TInt32(), null, null, null)

    assertShiftsTo(TInt64(), 5L, 2, 5L << 2)
    assertShiftsTo(TInt64(), -5L, 2, -5L << 2)
    assertShiftsTo(TInt64(), 5L, null, null)
    assertShiftsTo(TInt64(), null, 2, null)
    assertShiftsTo(TInt64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpRightShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(RightShift(), In(0, t), In(1, TInt32())), FastIndexedSeq(x -> t, y -> TInt32()), expected)
    }

    assertShiftsTo(TInt32(), 0xff5, 2, 0xff5 >> 2)
    assertShiftsTo(TInt32(), -5, 2, -5 >> 2)
    assertShiftsTo(TInt32(), 5, null, null)
    assertShiftsTo(TInt32(), null, 2, null)
    assertShiftsTo(TInt32(), null, null, null)

    assertShiftsTo(TInt64(), 0xffff5L, 2, 0xffff5L >> 2)
    assertShiftsTo(TInt64(), -5L, 2, -5L >> 2)
    assertShiftsTo(TInt64(), 5L, null, null)
    assertShiftsTo(TInt64(), null, 2, null)
    assertShiftsTo(TInt64(), null, null, null)
  }

  @Test def testApplyBinaryPrimOpLogicalRightShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(LogicalRightShift(), In(0, t), In(1, TInt32())), FastIndexedSeq(x -> t, y -> TInt32()), expected)
    }

    assertShiftsTo(TInt32(), 0xff5, 2, 0xff5 >>> 2)
    assertShiftsTo(TInt32(), -5, 2, -5 >>> 2)
    assertShiftsTo(TInt32(), 5, null, null)
    assertShiftsTo(TInt32(), null, 2, null)
    assertShiftsTo(TInt32(), null, null, null)

    assertShiftsTo(TInt64(), 0xffff5L, 2, 0xffff5L >>> 2)
    assertShiftsTo(TInt64(), -5L, 2, -5L >>> 2)
    assertShiftsTo(TInt64(), 5L, null, null)
    assertShiftsTo(TInt64(), null, 2, null)
    assertShiftsTo(TInt64(), null, null, null)
  }

  @Test def testApplyComparisonOpGT() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(GT(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, false)
    assertComparesTo(TInt32(), 0, 1, false)
    assertComparesTo(TInt32(), 1, 0, true)

    assertComparesTo(TInt64(), 1L, 1L, false)
    assertComparesTo(TInt64(), 0L, 1L, false)
    assertComparesTo(TInt64(), 1L, 0L, true)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, false)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, false)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, true)

    assertComparesTo(TFloat64(), 1.0, 1.0, false)
    assertComparesTo(TFloat64(), 0.0, 1.0, false)
    assertComparesTo(TFloat64(), 1.0, 0.0, true)

  }

  @Test def testApplyComparisonOpGTEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(GTEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, true)
    assertComparesTo(TInt32(), 0, 1, false)
    assertComparesTo(TInt32(), 1, 0, true)

    assertComparesTo(TInt64(), 1L, 1L, true)
    assertComparesTo(TInt64(), 0L, 1L, false)
    assertComparesTo(TInt64(), 1L, 0L, true)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, true)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, false)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, true)

    assertComparesTo(TFloat64(), 1.0, 1.0, true)
    assertComparesTo(TFloat64(), 0.0, 1.0, false)
    assertComparesTo(TFloat64(), 1.0, 0.0, true)
  }

  @Test def testApplyComparisonOpLT() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(LT(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, false)
    assertComparesTo(TInt32(), 0, 1, true)
    assertComparesTo(TInt32(), 1, 0, false)

    assertComparesTo(TInt64(), 1L, 1L, false)
    assertComparesTo(TInt64(), 0L, 1L, true)
    assertComparesTo(TInt64(), 1L, 0L, false)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, false)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, true)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, false)

    assertComparesTo(TFloat64(), 1.0, 1.0, false)
    assertComparesTo(TFloat64(), 0.0, 1.0, true)
    assertComparesTo(TFloat64(), 1.0, 0.0, false)

  }

  @Test def testApplyComparisonOpLTEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(LTEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, true)
    assertComparesTo(TInt32(), 0, 1, true)
    assertComparesTo(TInt32(), 1, 0, false)

    assertComparesTo(TInt64(), 1L, 1L, true)
    assertComparesTo(TInt64(), 0L, 1L, true)
    assertComparesTo(TInt64(), 1L, 0L, false)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, true)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, true)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, false)

    assertComparesTo(TFloat64(), 1.0, 1.0, true)
    assertComparesTo(TFloat64(), 0.0, 1.0, true)
    assertComparesTo(TFloat64(), 1.0, 0.0, false)

  }

  @Test def testApplyComparisonOpEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(EQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, expected = true)
    assertComparesTo(TInt32(), 0, 1, expected = false)
    assertComparesTo(TInt32(), 1, 0, expected = false)

    assertComparesTo(TInt64(), 1L, 1L, expected = true)
    assertComparesTo(TInt64(), 0L, 1L, expected = false)
    assertComparesTo(TInt64(), 1L, 0L, expected = false)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, expected = true)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, expected = false)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, expected = false)

    assertComparesTo(TFloat64(), 1.0, 1.0, expected = true)
    assertComparesTo(TFloat64(), 0.0, 1.0, expected = false)
    assertComparesTo(TFloat64(), 1.0, 0.0, expected = false)
  }

  @Test def testApplyComparisonOpNE() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(NEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertComparesTo(TInt32(), 1, 1, expected = false)
    assertComparesTo(TInt32(), 0, 1, expected = true)
    assertComparesTo(TInt32(), 1, 0, expected = true)

    assertComparesTo(TInt64(), 1L, 1L, expected = false)
    assertComparesTo(TInt64(), 0L, 1L, expected = true)
    assertComparesTo(TInt64(), 1L, 0L, expected = true)

    assertComparesTo(TFloat32(), 1.0f, 1.0f, expected = false)
    assertComparesTo(TFloat32(), 0.0f, 1.0f, expected = true)
    assertComparesTo(TFloat32(), 1.0f, 0.0f, expected = true)

    assertComparesTo(TFloat64(), 1.0, 1.0, expected = false)
    assertComparesTo(TFloat64(), 0.0, 1.0, expected = true)
    assertComparesTo(TFloat64(), 1.0, 0.0, expected = true)
  }

  @Test def testIf() {
    assertEvalsTo(If(True(), I32(5), I32(7)), 5)
    assertEvalsTo(If(False(), I32(5), I32(7)), 7)
    assertEvalsTo(If(NA(TBoolean()), I32(5), I32(7)), null)
    assertEvalsTo(If(True(), NA(TInt32()), I32(7)), null)
  }

  @Test def testIfWithDifferentRequiredness() {
    val t = TStruct(true, "foo" -> TStruct("bar" -> TArray(TInt32Required, required = true)))
    val value = Row(Row(FastIndexedSeq(1, 2, 3)))
    assertEvalsTo(
      If.unify(
        In(0, TBoolean()),
        In(1, t),
        MakeStruct(Seq("foo" -> MakeStruct(Seq("bar" -> ArrayRange(I32(0), I32(1), I32(1))))))),
      FastIndexedSeq((true, TBoolean()), (value, t)),
      value
    )
  }

  @Test def testLet() {
    assertEvalsTo(Let("v", I32(5), Ref("v", TInt32())), 5)
    assertEvalsTo(Let("v", NA(TInt32()), Ref("v", TInt32())), null)
    assertEvalsTo(Let("v", I32(5), NA(TInt32())), null)
    assertEvalsTo(ArrayMap(Let("v", I32(5), ArrayRange(0, Ref("v", TInt32()), 1)), "x", Ref("x", TInt32()) + I32(2)),
      FastIndexedSeq(2, 3, 4, 5, 6))
    assertEvalsTo(
      ArrayMap(Let("q", I32(2),
      ArrayMap(Let("v", Ref("q", TInt32()) + I32(3),
        ArrayRange(0, Ref("v", TInt32()), 1)),
        "x", Ref("x", TInt32()) + Ref("q", TInt32()))),
        "y", Ref("y", TInt32()) + I32(3)),
      FastIndexedSeq(5, 6, 7, 8, 9))
  }

  @Test def testMakeArray() {
    assertEvalsTo(MakeArray(FastSeq(I32(5), NA(TInt32()), I32(-3)), TArray(TInt32())), FastIndexedSeq(5, null, -3))
    assertEvalsTo(MakeArray(FastSeq(), TArray(TInt32())), FastIndexedSeq())
  }

  @Test def testMakeStruct() {
    assertEvalsTo(MakeStruct(FastSeq()), Row())
    assertEvalsTo(MakeStruct(FastSeq("a" -> NA(TInt32()), "b" -> 4, "c" -> 0.5)), Row(null, 4, 0.5))
    //making sure wide structs get emitted without failure
    assertEvalsTo(GetField(MakeStruct((0 until 20000).map(i => s"foo$i" -> I32(1))), "foo1"), 1)
  }

  @Test def testMakeArrayWithDifferentRequiredness(): Unit = {
    val t = TArray(TStruct("a" -> TInt32Required, "b" -> TArray(TInt32Optional, required = true)))
    val value = Row(2, FastIndexedSeq(1))
    assertEvalsTo(
      MakeArray.unify(
        Seq(NA(t.elementType.deepOptional()), In(0, t.elementType))
      ),
      FastIndexedSeq((value, t.elementType)),
      FastIndexedSeq(null, value)
    )
  }

  @Test def testMakeTuple() {
    assertEvalsTo(MakeTuple(FastSeq()), Row())
    assertEvalsTo(MakeTuple(FastSeq(NA(TInt32()), 4, 0.5)), Row(null, 4, 0.5))
    //making sure wide structs get emitted without failure
    assertEvalsTo(GetTupleElement(MakeTuple((0 until 20000).map(I32)), 1), 1)
  }

  @Test def testGetTupleElement() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = MakeTuple(FastIndexedSeq(I32(5), Str("abc"), NA(TInt32())))
    val na = NA(TTuple(TInt32(), TString()))

    assertEvalsTo(GetTupleElement(t, 0), 5)
    assertEvalsTo(GetTupleElement(t, 1), "abc")
    assertEvalsTo(GetTupleElement(t, 2), null)
    assertEvalsTo(GetTupleElement(na, 0), null)
  }

  @Test def testArrayRef() {
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(0)), 5)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), I32(1)), null)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32())), NA(TInt32())), null)

    assertFatal(ArrayRef(MakeArray(FastIndexedSeq(I32(5)), TArray(TInt32())), I32(2)), "array index out of bounds")
  }

  @Test def testArrayLen() {
    assertEvalsTo(ArrayLen(NA(TArray(TInt32()))), null)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(), TArray(TInt32()))), 0)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(I32(5), NA(TInt32())), TArray(TInt32()))), 2)
  }

  @Test def testArraySort() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ArraySort(NA(TArray(TInt32()))), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32()), I32(2)), TArray(TInt32()))
    assertEvalsTo(ArraySort(a),
      FastIndexedSeq(-7, 2, 2, null))
    assertEvalsTo(ArraySort(a, False()),
      FastIndexedSeq(2, 2, -7, null))
  }

  @Test def testToSet() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToSet(NA(TArray(TInt32()))), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32()), I32(2)), TArray(TInt32()))
    assertEvalsTo(ToSet(a), Set(-7, 2, null))
  }

  @Test def testToArrayFromSet() {
    val t = TSet(TInt32())
    assertEvalsTo(ToArray(NA(t)), null)
    assertEvalsTo(ToArray(In(0, t)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      FastIndexedSeq(-7, 2, null))
  }

  @Test def testToDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToDict(NA(TArray(TTuple(FastIndexedSeq(TInt32(), TString()))))), null)

    val a = MakeArray(FastIndexedSeq(
      MakeTuple(FastIndexedSeq(I32(5), Str("a"))),
      MakeTuple(FastIndexedSeq(I32(5), Str("a"))), // duplicate key-value pair
      MakeTuple(FastIndexedSeq(NA(TInt32()), Str("b"))),
      MakeTuple(FastIndexedSeq(I32(3), NA(TString()))),
      NA(TTuple(FastIndexedSeq(TInt32(), TString()))) // missing value
    ), TArray(TTuple(FastIndexedSeq(TInt32(), TString()))))

    assertEvalsTo(ToDict(a), Map(5 -> "a", (null, "b"), 3 -> null))
  }

  @Test def testToArrayFromDict() {
    val t = TDict(TInt32(), TString())
    assertEvalsTo(ToArray(NA(t)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(ToArray(In(0, t)),
      // wtf you can't do null -> ...
      FastIndexedSeq((d, t)),
      FastIndexedSeq(Row(1, "a"), Row(2, null), Row(null, "c")))
  }

  @Test def testToArrayFromArray() {
    val t = TArray(TInt32())
    assertEvalsTo(ToArray(NA(t)), null)
    assertEvalsTo(ToArray(In(0, t)),
      FastIndexedSeq((FastIndexedSeq(-7, 2, null, 2), t)),
      FastIndexedSeq(-7, 2, null, 2))
  }

  @Test def testSetContains() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TSet(TInt32())
    assertEvalsTo(invoke("contains", NA(t), I32(2)), null)

    assertEvalsTo(invoke("contains", In(0, t), NA(TInt32())),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(2)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(0)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      false)
    assertEvalsTo(invoke("contains", In(0, t), I32(7)),
      FastIndexedSeq((Set(-7, 2), t)),
      false)
  }

  @Test def testDictContains() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TDict(TInt32(), TString())
    assertEvalsTo(invoke("contains", NA(t), I32(2)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(invoke("contains", In(0, t), NA(TInt32())),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(2)),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", In(0, t), I32(0)),
      FastIndexedSeq((d, t)),
      false)
    assertEvalsTo(invoke("contains", In(0, t), I32(3)),
      FastIndexedSeq((Map(1 -> "a", 2 -> null), t)),
      false)
  }

  @Test def testLowerBoundOnOrderedCollectionArray() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TArray(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val awoutna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4)), TArray(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(-1), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(0), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(1), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(2), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(3), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(4), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, I32(5), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(awoutna, NA(TInt32()), onKey = false), 3)

    val awna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4), NA(TInt32())), TArray(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(awna, NA(TInt32()), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(awna, I32(5), onKey = false), 3)

    val awdups = MakeArray(FastIndexedSeq(I32(0), I32(0), I32(2), I32(4), I32(4), NA(TInt32())), TArray(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(awdups, I32(0), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(awdups, I32(4), onKey = false), 3)
  }

  @Test def testLowerBoundOnOrderedCollectionSet() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TSet(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val swoutna = ToSet(MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4), I32(4)), TArray(TInt32())))
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(-1), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(0), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(1), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(2), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(3), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(4), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(5), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, NA(TInt32()), onKey = false), 3)

    val swna = ToSet(MakeArray(FastIndexedSeq(I32(0), I32(2), I32(2), I32(4), NA(TInt32())), TArray(TInt32())))
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, NA(TInt32()), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, I32(5), onKey = false), 3)
  }

  @Test def testLowerBoundOnOrderedCollectionDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TDict(TInt32(), TString()))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = true), null)

    val dwna = TestUtils.IRDict((1, 3), (3, null), (null, 5))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(2), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(3), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(5), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, NA(TInt32()), onKey = true), 2)

    val dwoutna = TestUtils.IRDict((1, 3), (3, null))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(4), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, NA(TInt32()), onKey = true), 2)
  }

  @Test def testArrayMap() {
    val naa = NA(TArray(TInt32()))
    val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7)), TArray(TInt32()))

    assertEvalsTo(ArrayMap(naa, "a", I32(5)), null)

    assertEvalsTo(ArrayMap(a, "a", ApplyBinaryPrimOp(Add(), Ref("a", TInt32()), I32(1))), FastIndexedSeq(4, null, 8))

    assertEvalsTo(Let("a", I32(5),
      ArrayMap(a, "a", Ref("a", TInt32()))),
      FastIndexedSeq(3, null, 7))
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

  @Test def testArrayFlatMap() {
    val ta = TArray(TInt32())
    val taa = TArray(ta)
    val naa = NA(taa)
    val naaa = MakeArray(FastIndexedSeq(NA(ta), NA(ta)), taa)
    val a = MakeArray(FastIndexedSeq(
      MakeArray(FastIndexedSeq(I32(7), NA(TInt32())), ta),
      NA(ta),
      MakeArray(FastIndexedSeq(I32(2)), ta)),
      taa)

    assertEvalsTo(ArrayFlatMap(naa, "a", MakeArray(FastIndexedSeq(I32(5)), ta)), null)

    assertEvalsTo(ArrayFlatMap(naaa, "a", Ref("a", ta)), FastIndexedSeq())

    assertEvalsTo(ArrayFlatMap(a, "a", Ref("a", ta)), FastIndexedSeq(7, null, 2))

    assertEvalsTo(ArrayFlatMap(ArrayRange(I32(0), I32(3), I32(1)), "i", ArrayRef(a, Ref("i", TInt32()))), FastIndexedSeq(7, null, 2))

    assertEvalsTo(Let("a", I32(5), ArrayFlatMap(a, "a", Ref("a", ta))), FastIndexedSeq(7, null, 2))

    val arr = MakeArray(List(I32(1), I32(5), I32(2), NA(TInt32())), TArray(TInt32()))
    val expected = FastIndexedSeq(-1, 0, -1, 0, 1, 2, 3, 4, -1, 0, 1)
    assertEvalsTo(ArrayFlatMap(arr, "foo", ArrayRange(I32(-1), Ref("foo", TInt32()), I32(1))), expected)
  }

  @Test def testArrayFold() {
    def fold(array: IR, zero: IR, f: (IR, IR) => IR): IR =
      ArrayFold(array, zero, "_accum", "_elt", f(Ref("_accum", zero.typ), Ref("_elt", zero.typ)))

    assertEvalsTo(fold(ArrayRange(1, 2, 1), NA(TBoolean()), (accum, elt) => IsNA(accum)), true)
    assertEvalsTo(fold(TestUtils.IRArray(1, 2, 3), 0, (accum, elt) => accum + elt), 6)
    assertEvalsTo(fold(TestUtils.IRArray(1, 2, 3), NA(TInt32()), (accum, elt) => accum + elt), null)
    assertEvalsTo(fold(TestUtils.IRArray(1, null, 3), NA(TInt32()), (accum, elt) => accum + elt), null)
    assertEvalsTo(fold(TestUtils.IRArray(1, null, 3), 0, (accum, elt) => accum + elt), null)
  }

  @Test def testArrayScan() {
    implicit val execStrats = ExecStrategy.javaOnly

    def scan(array: IR, zero: IR, f: (IR, IR) => IR): IR =
      ArrayScan(array, zero, "_accum", "_elt", f(Ref("_accum", zero.typ), Ref("_elt", zero.typ)))

    assertEvalsTo(scan(ArrayRange(1, 4, 1), NA(TBoolean()), (accum, elt) => IsNA(accum)), FastIndexedSeq(null, true, false, false))
    assertEvalsTo(scan(TestUtils.IRArray(1, 2, 3), 0, (accum, elt) => accum + elt), FastIndexedSeq(0, 1, 3, 6))
    assertEvalsTo(scan(TestUtils.IRArray(1, 2, 3), NA(TInt32()), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(TestUtils.IRArray(1, null, 3), NA(TInt32()), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(TestUtils.IRArray(1, null, 3), 0, (accum, elt) => accum + elt), FastIndexedSeq(0, 1, null, null))
  }

  @Test def testNDArrayRef() {
    implicit val execStrats = Set(ExecStrategy.CxxCompile)

    def makeNDArray(data: Seq[Double], shape: Seq[Long], rowMajor: IR): MakeNDArray = {
      MakeNDArray(MakeArray(data.map(F64), TArray(TFloat64())), MakeArray(shape.map(I64), TArray(TInt64())), rowMajor)
    }
    def makeNDArrayRef(nd: IR, indxs: Seq[Long]): NDArrayRef = {
      NDArrayRef(nd, MakeArray(indxs.map(I64), TArray(TInt64())))
    }

    def scalarRowMajor = makeNDArray(FastSeq(3.0), FastSeq(), True())
    def scalarColMajor = makeNDArray(FastSeq(3.0), FastSeq(), False())
    assertEvalsTo(makeNDArrayRef(scalarRowMajor, FastSeq()), 3.0)
    assertEvalsTo(makeNDArrayRef(scalarColMajor, FastSeq()), 3.0)

    def vectorRowMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), True())
    def vectorColMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), False())
    assertEvalsTo(makeNDArrayRef(vectorRowMajor, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorColMajor, FastSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorRowMajor, FastSeq(1)), -1.0)
    assertEvalsTo(makeNDArrayRef(vectorColMajor, FastSeq(1)), -1.0)

    def threeTensorRowMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), True())
    def threeTensorColMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), False())
    def sevenRowMajor = makeNDArrayRef(threeTensorRowMajor, FastSeq(0, 1, 2))
    def sevenColMajor = makeNDArrayRef(threeTensorColMajor, FastSeq(1, 0, 1))
    assertEvalsTo(sevenRowMajor, 7.0)
    assertEvalsTo(sevenColMajor, 7.0)

    def cubeRowMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), True())
    def cubeColMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), False())
    def centerRowMajor = makeNDArrayRef(cubeRowMajor, FastSeq(1, 1, 1))
    def centerColMajor = makeNDArrayRef(cubeColMajor, FastSeq(1, 1, 1))
    assertEvalsTo(centerRowMajor, 13.0)
    assertEvalsTo(centerColMajor, 13.0)
  }

  @Test def testNDArrayMap() {
    implicit val execStrats = Set(ExecStrategy.CxxCompile)

    def data = 0 until 10
    def shape = MakeArray(FastSeq(2L, 5L).map(I64), TArray(TInt64()))

    def positives = MakeNDArray(MakeArray(data.map(i => F64(i.toDouble)), TArray(TFloat64())), shape, True())
    def negatives = NDArrayMap(positives, "e", ApplyUnaryPrimOp(Negate(), Ref("e", TFloat64())))
    assertEvalsTo(NDArrayRef(positives, MakeArray(FastSeq(1L, 0L), TArray(TInt64()))), 5.0)
    assertEvalsTo(NDArrayRef(negatives, MakeArray(FastSeq(1L, 0L), TArray(TInt64()))), -5.0)

    def trues = MakeNDArray(MakeArray(data.map(_ => True()), TArray(TBoolean())), shape, True())
    def falses = NDArrayMap(trues, "e", ApplyUnaryPrimOp(Bang(), Ref("e", TBoolean())))
    assertEvalsTo(NDArrayRef(trues, MakeArray(FastSeq(1L, 0L), TArray(TInt64()))), true)
    assertEvalsTo(NDArrayRef(falses, MakeArray(FastSeq(1L, 0L), TArray(TInt64()))), false)

    def bools = MakeNDArray(
      MakeArray(data.map(i => if (i % 2 == 0) True() else False()), TArray(TBoolean())), shape, False())
    def boolsToBinary = NDArrayMap(bools, "e", If(Ref("e", TBoolean()), I64(1L), I64(0L)))
    def one = NDArrayRef(boolsToBinary, MakeArray(FastSeq(0L, 0L), TArray(TInt64())))
    def zero = NDArrayRef(boolsToBinary, MakeArray(FastSeq(1L, 0L), TArray(TInt64())))
    assertEvalsTo(one, 1L)
    assertEvalsTo(zero, 0L)
  }

  @Test def testLeftJoinRightDistinct() {
    implicit val execStrats = ExecStrategy.javaOnly

    def join(left: IR, right: IR, keys: IndexedSeq[String]): IR = {
      val compF = { (l: IR, r: IR) =>
        ApplyComparisonOp(Compare(coerce[TStruct](l.typ).select(keys)._1), SelectFields(l, keys), SelectFields(r, keys))
      }
      val joinF = { (l: IR, r: IR) =>
        Let("_right", r, InsertFields(l, coerce[TStruct](r.typ).fields.filter(f => !keys.contains(f.name)).map { f =>
          f.name -> GetField(Ref("_right", r.typ), f.name)
        }))
      }
      ArrayLeftJoinDistinct(left, right, "_l", "_r",
        compF(Ref("_l", coerce[TArray](left.typ).elementType), Ref("_r", coerce[TArray](right.typ).elementType)),
        joinF(Ref("_l", coerce[TArray](left.typ).elementType), Ref("_r", coerce[TArray](right.typ).elementType)))
    }

    def joinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR = {
      join(
        MakeArray.unify(left.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("k1" -> (if (n == null) NA(TInt32()) else I32(n)), "k2" -> Str("x"), "a" -> I64(idx))) }),
        MakeArray.unify(right.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("b" -> I32(idx), "k2" -> Str("x"), "k1" -> (if (n == null) NA(TInt32()) else I32(n)), "c" -> Str("foo"))) }),
        FastIndexedSeq("k1", "k2"))
    }

    assertEvalsTo(joinRows(Array[Integer](0, null), Array[Integer](1, null)), FastIndexedSeq(
      Row(0, "x", 0L, null, null),
      Row(null, "x", 1L, 1, "foo")))

    assertEvalsTo(joinRows(Array[Integer](0, 1, 2), Array[Integer](1)), FastIndexedSeq(
      Row(0, "x", 0L, null, null),
      Row(1, "x", 1L, 0, "foo"),
      Row(2, "x", 2L, null, null)))

    assertEvalsTo(joinRows(Array[Integer](0, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)), FastIndexedSeq(
      Row(0, "x", 0L, 1, "foo"),
      Row(1, "x", 1L, 3, "foo"),
      Row(2, "x", 2L, 5, "foo")))

    assertEvalsTo(joinRows(Array[Integer](0, 1, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)), FastIndexedSeq(
      Row(0, "x", 0L, 1, "foo"),
      Row(1, "x", 1L, 3, "foo"),
      Row(1, "x", 2L, 3, "foo"),
      Row(2, "x", 3L, 5, "foo")))
  }

  @Test def testDie() {
    assertFatal(Die("mumblefoo", TFloat64()), "mble")
    assertFatal(Die(NA(TString()), TFloat64()), "message missing")
  }

  @Test def testArrayRange() {
    def assertEquals(start: Integer, stop: Integer, step: Integer, expected: IndexedSeq[Int]) {
      assertEvalsTo(ArrayRange(In(0, TInt32()), In(1, TInt32()), In(2, TInt32())),
        args = FastIndexedSeq(start -> TInt32(), stop -> TInt32(), step -> TInt32()),
        expected = expected)
    }
    assertEquals(0, 5, null, null)
    assertEquals(0, null, 1, null)
    assertEquals(null, 5, 1, null)

    assertFatal(ArrayRange(I32(0), I32(5), I32(0)), "step size")

    for {
      start <- -2 to 2
      stop <- -2 to 8
      step <- 1 to 3
    } {
      assertEquals(start, stop, step, expected = Array.range(start, stop, step).toFastIndexedSeq)
      assertEquals(start, stop, -step, expected = Array.range(start, stop, -step).toFastIndexedSeq)
    }
    // this needs to be written this way because of a bug in Scala's Array.range
    val expected = Array.tabulate(11)(Int.MinValue + _ * (Int.MaxValue / 5)).toFastIndexedSeq
    assertEquals(Int.MinValue, Int.MaxValue, Int.MaxValue / 5, expected)
  }

  @Test def testArrayAgg() {
    implicit val execStrats = ExecStrategy.javaOnly

    val sumSig = AggSignature(Sum(), Seq(), None, Seq(TInt64()))
    assertEvalsTo(
      ArrayAgg(
        ArrayMap(ArrayRange(I32(0), I32(4), I32(1)), "x", Cast(Ref("x", TInt32()), TInt64())),
        "x",
        ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq(Ref("x", TInt64())), sumSig)),
      6L)
  }

  @Test def testInsertFields() {
    implicit val execStrats = ExecStrategy.javaOnly

    val s = TStruct("a" -> TInt64(), "b" -> TString())
    val emptyStruct = MakeStruct(Seq("a" -> NA(TInt64()), "b" -> NA(TString())))

    assertEvalsTo(
      InsertFields(
        NA(s),
        Seq()),
      null)

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        Seq("a" -> I64(5))),
      Row(5L, null))

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        Seq("c" -> F64(3.2))),
      Row(null, null, 3.2))

    assertEvalsTo(
      InsertFields(
        emptyStruct,
        Seq("c" -> NA(TFloat64()))),
      Row(null, null, null))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64()), "b" -> Str("abc"))),
        Seq()),
      Row(null, "abc"))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64()), "b" -> Str("abc"))),
        Seq("a" -> I64(5))),
      Row(5L, "abc"))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64()), "b" -> Str("abc"))),
        Seq("c" -> F64(3.2))),
      Row(null, "abc", 3.2))

    assertEvalsTo(
      InsertFields(NA(TStruct("a" -> +TInt32())), Seq("foo" -> I32(5))),
      null
    )

    assertEvalsTo(
      InsertFields(
        In(0, s),
        Seq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastIndexedSeq("c", "d", "e", "a", "b"))),
      FastIndexedSeq(Row(null, "abc") -> s),
      Row(3.2, 5.5, 6.6, null, "abc"))

    assertEvalsTo(
      InsertFields(
        In(0, s),
        Seq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastIndexedSeq("a", "b", "c", "d", "e"))),
      FastIndexedSeq(Row(null, "abc") -> s),
      Row(null, "abc", 3.2, 5.5, 6.6))

    assertEvalsTo(
      InsertFields(
        In(0, s),
        Seq("c" -> F64(3.2), "d" -> F64(5.5), "e" -> F64(6.6)),
        Some(FastIndexedSeq("c", "a", "d", "b", "e"))),
      FastIndexedSeq(Row(null, "abc") -> s),
      Row(3.2, null, 5.5, "abc", 6.6))

  }

  @Test def testSelectFields() {
    assertEvalsTo(
      SelectFields(
        NA(TStruct("foo" -> TInt32(), "bar" -> TFloat64())),
        FastSeq("foo")),
      null)

    assertEvalsTo(
      SelectFields(
        MakeStruct(FastSeq("foo" -> 6, "bar" -> 0.0)),
        FastSeq("foo")),
      Row(6))

    assertEvalsTo(
      SelectFields(
        MakeStruct(FastSeq("a" -> 6, "b" -> 0.0, "c" -> 3L)),
        FastSeq("b", "a")),
      Row(0.0, 6))

  }

  @Test def testGetField() {
    implicit val execStrats = ExecStrategy.javaOnly

    val s = MakeStruct(Seq("a" -> NA(TInt64()), "b" -> Str("abc")))
    val na = NA(TStruct("a" -> TInt64(), "b" -> TString()))

    assertEvalsTo(GetField(s, "a"), null)
    assertEvalsTo(GetField(s, "b"), "abc")
    assertEvalsTo(GetField(na, "a"), null)
  }

  @Test def testLiteral() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.CxxCompile)
    val poopEmoji = new String(Array[Char](0xD83D, 0xDCA9))
    val types = Array(
      TTuple(TInt32(), TString(), TArray(TInt32())),
      TArray(TString()),
      TDict(TInt32(), TString())
    )
    val values = Array(
      Row(400, "foo"+poopEmoji, FastIndexedSeq(4, 6, 8)),
      FastIndexedSeq(poopEmoji, "", "foo"),
      Map[Int, String](1 -> "", 5 -> "foo", -4 -> poopEmoji)
    )

    assertEvalsTo(Literal(types(0), values(0)), values(0))
    assertEvalsTo(MakeTuple(types.zip(values).map { case (t, v) => Literal(t, v) }), Row.fromSeq(values.toFastSeq))
    assertEvalsTo(Str("hello"+poopEmoji), "hello"+poopEmoji)
  }

  @Test def testTableCount() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.CxxCompile)
    assertEvalsTo(TableCount(TableRange(0, 4)), 0L)
    assertEvalsTo(TableCount(TableRange(7, 4)), 7L)
  }

  @Test def testTableGetGlobals() {
    implicit val execStrats = ExecStrategy.interpretOnly
    assertEvalsTo(TableGetGlobals(TableMapGlobals(TableRange(0, 1), Literal(TStruct("a" -> TInt32()), Row(1)))), Row(1))
  }

  @Test def testTableAggregate() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val table = Table.range(hc, 3, Some(2))
    val countSig = AggSignature(Count(), Seq(), None, Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq.empty, countSig)
    assertEvalsTo(TableAggregate(table.tir, MakeStruct(Seq("foo" -> count))), Row(3L))
  }

  @Test def testMatrixAggregate() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val matrix = MatrixTable.range(hc, 5, 5, None)
    val countSig = AggSignature(Count(), Seq(), None, Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq.empty, countSig)
    assertEvalsTo(MatrixAggregate(matrix.ast, MakeStruct(Seq("foo" -> count))), Row(25L))
  }

  @Test def testGroupByKey() {
    implicit val execStrats = ExecStrategy.javaOnly

    def tuple(k: String, v: Int): IR = MakeTuple(Seq(Str(k), I32(v)))

    def groupby(tuples: IR*): IR = GroupByKey(MakeArray(tuples, TArray(TTuple(TString(), TInt32()))))

    val collection1 = groupby(tuple("foo", 0), tuple("bar", 4), tuple("foo", -1), tuple("bar", 0), tuple("foo", 10), tuple("", 0))

    assertEvalsTo(collection1, Map("" -> FastIndexedSeq(0), "bar" -> FastIndexedSeq(4, 0), "foo" -> FastIndexedSeq(0, -1, 10)))
  }

  @DataProvider(name = "compareDifferentTypes")
  def compareDifferentTypesData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(0.0, 0.0), TArray(+TFloat64()), TArray(TFloat64())),
    Array(Set(0, 1), TSet(+TInt32()), TSet(TInt32())),
    Array(Map(0L -> 5, 3L -> 20), TDict(+TInt64(), TInt32()), TDict(TInt64(), +TInt32())),
    Array(Interval(1, 2, includesStart = false, includesEnd = true), TInterval(+TInt32()), TInterval(TInt32())),
    Array(Row("foo", 0.0), TStruct("a" -> +TString(), "b" -> +TFloat64()), TStruct("a" -> TString(), "b" -> TFloat64())),
    Array(Row("foo", 0.0), TTuple(TString(), +TFloat64()), TTuple(+TString(), +TFloat64())),
    Array(Row(FastIndexedSeq("foo"), 0.0), TTuple(+TArray(TString()), +TFloat64()), TTuple(TArray(+TString()), +TFloat64()))
  )

  @Test(dataProvider = "compareDifferentTypes")
  def testComparisonOpDifferentTypes(a: Any, t1: Type, t2: Type) {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ApplyComparisonOp(EQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(LT(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(GT(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(LTEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(GTEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQ(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(EQWithNA(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQWithNA(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(Compare(t1, t2), In(0, t1), In(1, t2)), IndexedSeq(a -> t1, a -> t2), 0)
  }

  @DataProvider(name = "valueIRs")
  def valueIRs(): Array[Array[IR]] = {
    hc.indexBgen(FastIndexedSeq("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))

    val b = True()
    val c = Ref("c", TBoolean())
    val i = I32(5)
    val j = I32(7)
    val str = Str("Hail")
    val a = Ref("a", TArray(TInt32()))
    val aa = Ref("aa", TArray(TArray(TInt32())))
    val da = Ref("da", TArray(TTuple(TInt32(), TString())))
    val nd = Ref("nd", TNDArray(TFloat64()))
    val v = Ref("v", TInt32())
    val s = Ref("s", TStruct("x" -> TInt32(), "y" -> TInt64(), "z" -> TFloat64()))
    val t = Ref("t", TTuple(TInt32(), TInt64(), TFloat64()))

    val call = Ref("call", TCall())

    val collectSig = AggSignature(Collect(), Seq(), None, Seq(TInt32()))

    val sumSig = AggSignature(Sum(), Seq(), None, Seq(TInt32()))

    val callStatsSig = AggSignature(CallStats(), Seq(), Some(Seq(TInt32())), Seq(TCall()))

    val histSig = AggSignature(Histogram(), Seq(TFloat64(), TFloat64(), TInt32()), None, Seq(TFloat64()))

    val takeBySig = AggSignature(TakeBy(), Seq(TInt32()), None, Seq(TFloat64(), TInt32()))

    val countSig = AggSignature(Count(), Seq(), None, Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq.empty, countSig)

    val table = TableRange(100, 10)

    val mt = MatrixTable.range(hc, 20, 2, Some(3)).ast.asInstanceOf[MatrixRead]
    val vcf = is.hail.TestUtils.importVCF(hc, "src/test/resources/sample.vcf")
      .ast.asInstanceOf[MatrixRead]

    val bgenReader = MatrixBGENReader(FastIndexedSeq("src/test/resources/example.8bits.bgen"), None, Map.empty[String, String], None, None, None)
    val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

    val blockMatrix = BlockMatrixRead(BlockMatrixNativeReader(tmpDir.createLocalTempFile()))

    val irs = Array(
      i, I64(5), F32(3.14f), F64(3.14), str, True(), False(), Void(),
      Cast(i, TFloat64()),
      NA(TInt32()), IsNA(i),
      If(b, i, j),
      Let("v", i, v),
      AggLet("v", i, v, false),
      Ref("x", TInt32()),
      ApplyBinaryPrimOp(Add(), i, j),
      ApplyUnaryPrimOp(Negate(), i),
      ApplyComparisonOp(EQ(TInt32()), i, j),
      MakeArray(FastSeq(i, NA(TInt32()), I32(-3)), TArray(TInt32())),
      MakeNDArray(
        MakeArray(FastSeq(F64(-1.0), F64(1.0)), TArray(TFloat64())),
        MakeArray(FastSeq(I64(1), I64(2)), TArray(TInt64())),
        True()),
      NDArrayRef(nd, MakeArray(FastSeq(I64(1), I64(2)), TArray(TInt64()))),
      NDArrayMap(nd, "v", ApplyUnaryPrimOp(Negate(), v)),
      ArrayRef(a, i),
      ArrayLen(a),
      ArrayRange(I32(0), I32(5), I32(1)),
      ArraySort(a, b),
      ToSet(a),
      ToDict(da),
      ToArray(a),
      ToStream(a),
      LowerBoundOnOrderedCollection(a, i, onKey = true),
      GroupByKey(da),
      ArrayMap(a, "v", v),
      ArrayFilter(a, "v", b),
      ArrayFlatMap(aa, "v", a),
      ArrayFold(a, I32(0), "x", "v", v),
      ArrayScan(a, I32(0), "x", "v", v),
      ArrayLeftJoinDistinct(ArrayRange(0, 2, 1), ArrayRange(0, 3, 1), "l", "r", I32(0), I32(1)),
      ArrayFor(a, "v", Void()),
      ArrayAgg(a, "x", ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq(Ref("x", TInt32())), sumSig)),
      AggFilter(True(), I32(0), false),
      AggExplode(NA(TArray(TInt32())), "x", I32(0), false),
      AggGroupBy(True(), I32(0), false),
      ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq(I32(0)), collectSig),
      ApplyAggOp(FastIndexedSeq(F64(-5.0), F64(5.0), I32(100)), None, FastIndexedSeq(F64(-2.11)), histSig),
      ApplyAggOp(FastIndexedSeq.empty, Some(FastIndexedSeq(I32(2))), FastIndexedSeq(call), callStatsSig),
      ApplyAggOp(FastIndexedSeq(I32(10)), None, FastIndexedSeq(F64(-2.11), I32(4)), takeBySig),
      InitOp(I32(0), FastIndexedSeq(I32(2)), callStatsSig),
      SeqOp(I32(0), FastIndexedSeq(i), collectSig),
      SeqOp(I32(0), FastIndexedSeq(F64(-2.11), I32(17)), takeBySig),
      Begin(IndexedSeq(Void())),
      MakeStruct(Seq("x" -> i)),
      SelectFields(s, Seq("x", "z")),
      InsertFields(s, Seq("x" -> i)),
      InsertFields(s, Seq("* x *" -> i)), // Won't parse as a simple identifier
      GetField(s, "x"),
      MakeTuple(Seq(i, b)),
      GetTupleElement(t, 1),
      In(2, TFloat64()),
      Die("mumblefoo", TFloat64()),
      invoke("&&", b, c), // ApplySpecial
      invoke("toFloat64", i), // Apply
      Uniroot("x", F64(3.14), F64(-5.0), F64(5.0)),
      Literal(TStruct("x" -> TInt32()), Row(1)),
      TableCount(table),
      TableGetGlobals(table),
      TableCollect(table),
      TableAggregate(table, MakeStruct(Seq("foo" -> count))),
      TableToValueApply(table, ForceCountTable()),
      MatrixToValueApply(mt, ForceCountMatrixTable()),
      TableWrite(table, tmpDir.createLocalTempFile(extension = "ht")),
      TableExport(table, tmpDir.createLocalTempFile(extension = "tsv"), null, true, ExportType.CONCATENATED, ","),
      MatrixWrite(mt, MatrixNativeWriter(tmpDir.createLocalTempFile(extension = "mt"))),
      MatrixWrite(vcf, MatrixVCFWriter(tmpDir.createLocalTempFile(extension = "vcf"))),
      MatrixWrite(vcf, MatrixPLINKWriter(tmpDir.createLocalTempFile())),
      MatrixWrite(bgen, MatrixGENWriter(tmpDir.createLocalTempFile())),
      MatrixMultiWrite(Array(mt, mt), MatrixNativeMultiWriter(tmpDir.createLocalTempFile())),
      MatrixAggregate(mt, MakeStruct(Seq("foo" -> count))),
      BlockMatrixWrite(blockMatrix, BlockMatrixNativeWriter(tmpDir.createLocalTempFile(), false, false, false)),
      CollectDistributedArray(ArrayRange(0, 3, 1), 1, "x", "y", Ref("x", TInt32())),
      ReadPartition(Str("foo"), CodecSpec.default, TStruct("foo"->TInt32(), "bar" -> TString()), TStruct("foo"->TInt32()))
    )
    irs.map(x => Array(x))
  }

  @DataProvider(name = "tableIRs")
  def tableIRs(): Array[Array[TableIR]] = {
    try {
      val ht = Table.read(hc, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
      val mt = MatrixTable.read(hc, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")

      val read = ht.tir.asInstanceOf[TableRead]
      val mtRead = mt.ast.asInstanceOf[MatrixRead]
      val b = True()

      val xs: Array[TableIR] = Array(
        TableDistinct(read),
        TableKeyBy(read, Array("m", "d")),
        TableFilter(read, b),
        read,
        MatrixColsTable(mtRead),
        TableAggregateByKey(read,
          MakeStruct(FastIndexedSeq(
            "a" -> I32(5)))),
        TableKeyByAndAggregate(read,
          NA(TStruct()), NA(TStruct()), Some(1), 2),
        TableJoin(read,
          TableRange(100, 10), "inner", 1),
        TableLeftJoinRightDistinct(read, TableRange(100, 10), "root"),
        TableMultiWayZipJoin(IndexedSeq(read, read), " * data * ", "globals"),
        MatrixEntriesTable(mtRead),
        MatrixRowsTable(mtRead),
        TableRepartition(read, 10, RepartitionStrategy.COALESCE),
        TableHead(read, 10),
        TableParallelize(
          MakeStruct(FastSeq(
            "rows" -> MakeArray(FastSeq(
            MakeStruct(FastSeq("a" -> NA(TInt32()))),
            MakeStruct(FastSeq("a" -> I32(1)))
          ), TArray(TStruct("a" -> TInt32()))),
            "global" -> MakeStruct(FastSeq()))), None),
        TableMapRows(TableKeyBy(read, FastIndexedSeq()),
          MakeStruct(FastIndexedSeq(
            "a" -> GetField(Ref("row", read.typ.rowType), "f32"),
            "b" -> F64(-2.11)))),
        TableMapGlobals(read,
          MakeStruct(FastIndexedSeq(
            "foo" -> NA(TArray(TInt32()))))),
        TableRange(100, 10),
        TableUnion(
          FastIndexedSeq(TableRange(100, 10), TableRange(50, 10))),
        TableExplode(read, Array("mset")),
        TableOrderBy(TableKeyBy(read, FastIndexedSeq()), FastIndexedSeq(SortField("m", Ascending), SortField("m", Descending))),
        CastMatrixToTable(mtRead, " # entries", " # cols"),
        TableRename(read, Map("idx" -> "idx_foo"), Map("global_f32" -> "global_foo"))
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
  def matrixIRs(): Array[Array[MatrixIR]] = {
    try {
      hc.indexBgen(FastIndexedSeq("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))

      val tableRead = Table.read(hc, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
        .tir.asInstanceOf[TableRead]
      val read = MatrixTable.read(hc, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")
        .ast.asInstanceOf[MatrixRead]
      val range = MatrixTable.range(hc, 3, 7, None)
        .ast.asInstanceOf[MatrixRead]
      val vcf = is.hail.TestUtils.importVCF(hc, "src/test/resources/sample.vcf")
        .ast.asInstanceOf[MatrixRead]

      val bgenReader = MatrixBGENReader(FastIndexedSeq("src/test/resources/example.8bits.bgen"), None, Map.empty[String, String], None, None, None)
      val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

      val range1 = MatrixTable.range(hc, 20, 2, Some(3))
        .ast.asInstanceOf[MatrixRead]
      val range2 = MatrixTable.range(hc, 20, 2, Some(4))
        .ast.asInstanceOf[MatrixRead]

      val b = True()

      val newCol = MakeStruct(FastIndexedSeq(
        "col_idx" -> GetField(Ref("sa", read.typ.colType), "col_idx"),
        "new_f32" -> ApplyBinaryPrimOp(Add(),
          GetField(Ref("sa", read.typ.colType), "col_f32"),
          F32(-5.2f))))
      val newRow = MakeStruct(FastIndexedSeq(
        "row_idx" -> GetField(Ref("va", read.typ.rvRowType), "row_idx"),
        "new_f32" -> ApplyBinaryPrimOp(Add(),
          GetField(Ref("va", read.typ.rvRowType), "row_f32"),
          F32(-5.2f)))
      )

      val collectSig = AggSignature(Collect(), Seq(), None, Seq(TInt32()))
      val collect = ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq(I32(0)), collectSig)

      val newRowAnn = MakeStruct(FastIndexedSeq("count_row" -> collect))
      val newColAnn = MakeStruct(FastIndexedSeq("count_col" -> collect))
      val newEntryAnn = MakeStruct(FastIndexedSeq("count_entry" -> collect))

      val xs = Array[MatrixIR](
        read,
        MatrixFilterRows(read, b),
        MatrixFilterCols(read, b),
        MatrixFilterEntries(read, b),
        MatrixChooseCols(read, Array(0, 0, 0)),
        MatrixMapCols(read, newCol, None),
        MatrixKeyRowsBy(read, FastIndexedSeq("row_m", "row_d"), false),
        MatrixMapRows(read, newRow),
        MatrixRepartition(read, 10, 0),
        MatrixMapEntries(read, MakeStruct(FastIndexedSeq(
          "global_f32" -> ApplyBinaryPrimOp(Add(),
            GetField(Ref("global", read.typ.globalType), "global_f32"),
            F32(-5.2f))))),
        MatrixCollectColsByKey(read),
        MatrixAggregateColsByKey(read, newEntryAnn, newColAnn),
        MatrixAggregateRowsByKey(read, newEntryAnn, newRowAnn),
        range,
        vcf,
        bgen,
        MatrixExplodeRows(read, FastIndexedSeq("row_mset")),
        MatrixUnionRows(FastIndexedSeq(range1, range2)),
        MatrixDistinctByRow(range1),
        MatrixRowsHead(range1, 3),
        MatrixExplodeCols(read, FastIndexedSeq("col_mset")),
        CastTableToMatrix(
          CastMatrixToTable(read, " # entries", " # cols"),
          " # entries",
          " # cols",
          read.typ.colKey),
        MatrixAnnotateColsTable(read, tableRead, "uid_123"),
        MatrixAnnotateRowsTable(read, tableRead, "uid_123"),
        MatrixRename(read, Map("global_i64" -> "foo"), Map("col_i64" -> "bar"), Map("row_i64" -> "baz"), Map("entry_i64" -> "quam"))
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
    val read = BlockMatrixRead(BlockMatrixNativeReader("src/test/resources/blockmatrix_example/0"))
    val transpose = BlockMatrixBroadcast(read, IndexedSeq(1, 0), IndexedSeq(2, 2), 2)
    val dot = BlockMatrixDot(read, transpose)

    val blockMatrixIRs = Array[BlockMatrixIR](read, transpose, dot)

    blockMatrixIRs.map(ir => Array(ir))
  }

  @Test(dataProvider = "valueIRs")
  def testValueIRParser(x: IR) {
    val env = IRParserEnvironment(refMap = Map(
      "c" -> TBoolean(),
      "a" -> TArray(TInt32()),
      "aa" -> TArray(TArray(TInt32())),
      "da" -> TArray(TTuple(TInt32(), TString())),
      "nd" -> TNDArray(TFloat64()),
      "nd2" -> TNDArray(TArray(TString())),
      "v" -> TInt32(),
      "s" -> TStruct("x" -> TInt32(), "y" -> TInt64(), "z" -> TFloat64()),
      "t" -> TTuple(TInt32(), TInt64(), TFloat64()),
      "call" -> TCall(),
      "x" -> TInt32()
    ))

    val s = Pretty(x)
    val x2 = IRParser.parse_value_ir(s, env)
    assert(x2 == x)
  }

  @Test(dataProvider = "tableIRs")
  def testTableIRParser(x: TableIR) {
    val s = Pretty(x)
    val x2 = IRParser.parse_table_ir(s)
    assert(x2 == x)
  }

  @Test(dataProvider = "matrixIRs")
  def testMatrixIRParser(x: MatrixIR) {
    val s = Pretty(x)
    val x2 = IRParser.parse_matrix_ir(s)
    assert(x2 == x)
  }

  @Test(dataProvider = "blockMatrixIRs")
  def testBlockMatrixIRParser(x: BlockMatrixIR) {
    val s = Pretty(x)
    val x2 = IRParser.parse_blockmatrix_ir(s)
    assert(x2 == x)
  }

  @Test def testCachedIR() {
    val cached = Literal(TSet(TInt32()), Set(1))
    val s = s"(JavaIR __uid1)"
    val x2 = IRParser.parse_value_ir(s, IRParserEnvironment(refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    assert(x2 eq cached)
  }

  @Test def testCachedTableIR() {
    val cached = TableRange(1, 1)
    val s = s"(JavaTable __uid1)"
    val x2 = IRParser.parse_table_ir(s, IRParserEnvironment(refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    assert(x2 eq cached)
  }

  @Test def testCachedMatrixIR() {
    val cached = MatrixTable.range(hc, 3, 7, None).ast
    val s = s"(JavaMatrix __uid1)"
    val x2 = IRParser.parse_matrix_ir(s, IRParserEnvironment(refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    assert(x2 eq cached)
  }

  @Test def testCachedBlockMatrixIR() {
    val cached = new BlockMatrixLiteral(BlockMatrix.fill(hc, 3, 7, 1))
    val s = s"(JavaBlockMatrix __uid1)"
    val x2 = IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    assert(x2 eq cached)
  }

  @Test def testContextSavedMatrixIR() {
    val cached = MatrixTable.range(hc, 3, 8, None).ast
    val id = hc.addIrVector(Array(cached))
    val s = s"(JavaMatrixVectorRef $id 0)"
    val x2 = IRParser.parse_matrix_ir(s, IRParserEnvironment(refMap = Map.empty, irMap = Map.empty))
    assert(cached eq x2)

    is.hail.HailContext.pyRemoveIrVector(id)
    assert(hc.irVectors.get(id) eq None)
  }

  @Test def testEvaluations() {
    TestFunctions.registerAll()

    def test(x: IR, i: java.lang.Boolean, expectedEvaluations: Int) {
      val env = Env.empty[(Any, Type)]
      val args = IndexedSeq((i, TBoolean()))

      IRSuite.globalCounter = 0
      Interpret[Any](x, env, args, None, optimize = false)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      Interpret[Any](x, env, args, None)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      eval(x, env, args, None)
      assert(IRSuite.globalCounter == expectedEvaluations)
    }

    def i = In(0, TBoolean())

    def st = ApplySeeded("incr_s", FastSeq(True()), 0L)

    def sf = ApplySeeded("incr_s", FastSeq(True()), 0L)

    def sm = ApplySeeded("incr_s", FastSeq(NA(TBoolean())), 0L)

    def mt = ApplySeeded("incr_m", FastSeq(True()), 0L)

    def mf = ApplySeeded("incr_m", FastSeq(True()), 0L)

    def mm = ApplySeeded("incr_m", FastSeq(NA(TBoolean())), 0L)

    def vt = ApplySeeded("incr_v", FastSeq(True()), 0L)

    def vf = ApplySeeded("incr_v", FastSeq(True()), 0L)

    def vm = ApplySeeded("incr_v", FastSeq(NA(TBoolean())), 0L)

    // baseline
    test(st, true, 1); test(sf, true, 1); test(sm, true, 1)
    test(mt, true, 1); test(mf, true, 1); test(mm, true, 1)
    test(vt, true, 1); test(vf, true, 1); test(vm, true, 0)

    // if
    // condition
    test(If(st, i, True()), true, 1)
    test(If(sf, i, True()), true, 1)
    test(If(sm, i, True()), true, 1)

    test(If(mt, i, True()), true, 1)
    test(If(mf, i, True()), true, 1)
    test(If(mm, i, True()), true, 1)

    test(If(vt, i, True()), true, 1)
    test(If(vf, i, True()), true, 1)
    test(If(vm, i, True()), true, 0)

    // consequent
    test(If(i, st, True()), true, 1)
    test(If(i, sf, True()), true, 1)
    test(If(i, sm, True()), true, 1)

    test(If(i, mt, True()), true, 1)
    test(If(i, mf, True()), true, 1)
    test(If(i, mm, True()), true, 1)

    test(If(i, vt, True()), true, 1)
    test(If(i, vf, True()), true, 1)
    test(If(i, vm, True()), true, 0)

    // alternate
    test(If(i, True(), st), false, 1)
    test(If(i, True(), sf), false, 1)
    test(If(i, True(), sm), false, 1)

    test(If(i, True(), mt), false, 1)
    test(If(i, True(), mf), false, 1)
    test(If(i, True(), mm), false, 1)

    test(If(i, True(), vt), false, 1)
    test(If(i, True(), vf), false, 1)
    test(If(i, True(), vm), false, 0)
  }

  @Test def testArrayContinuationDealsWithIfCorrectly() {
    val ir = ArrayMap(
      If(IsNA(In(0, TBoolean())),
        NA(TArray(TInt32())),
        In(1, TArray(TInt32()))),
      "x", Cast(Ref("x", TInt32()), TInt64()))

    val env = Env.empty[(Any, Type)]
      .bind("flag" -> (true, TBoolean()))
      .bind("array" -> (FastIndexedSeq(0), TArray(TInt32())))

    assertEvalsTo(ir, FastIndexedSeq(true -> TBoolean(), FastIndexedSeq(0) -> TArray(TInt32())), FastIndexedSeq(0L))
  }

  @Test def setContainsSegfault(): Unit = {
    hc // assert initialized
    val irStr =
      """
        |(TableFilter
        |  (TableMapRows
        |    (TableKeyBy () False
        |      (TableMapRows
        |        (TableKeyBy () False
        |          (TableMapRows
        |            (TableRange 1 12)
        |            (InsertFields
        |              (Ref row)
        |              None
        |              (s
        |                (Literal Set[String] "[\"foo\"]"))
        |              (nested
        |                (NA Struct{elt:String})))))
        |        (InsertFields
        |          (Ref row) None)))
        |    (SelectFields (s nested)
        |      (Ref row)))
        |  (Let __uid_1
        |    (If
        |      (IsNA
        |        (GetField s
        |          (Ref row)))
        |      (NA Boolean)
        |      (Let __iruid_1
        |        (LowerBoundOnOrderedCollection False
        |          (GetField s
        |            (Ref row))
        |          (GetField elt
        |            (GetField nested
        |              (Ref row))))
        |        (If
        |          (ApplyComparisonOp EQ
        |            (Ref __iruid_1)
        |            (ArrayLen
        |              (ToArray
        |                (GetField s
        |                  (Ref row)))))
        |          (False)
        |          (ApplyComparisonOp EQ
        |            (ArrayRef
        |              (ToArray
        |                (GetField s
        |                  (Ref row)))
        |              (Ref __iruid_1))
        |            (GetField elt
        |              (GetField nested
        |                (Ref row)))))))
        |    (If
        |      (IsNA
        |        (Ref __uid_1))
        |      (False)
        |      (Ref __uid_1))))
      """.stripMargin

    Interpret(ir.IRParser.parse_table_ir(irStr), optimize = false).rvd.count()
  }

  @Test def testTableGetGlobalsSimplifyRules() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t1 = TableType(TStruct("a" -> TInt32()), IndexedSeq("a"), TStruct("g1" -> TInt32(), "g2" -> TFloat64()))
    val t2 = TableType(TStruct("a" -> TInt32()), IndexedSeq("a"), TStruct("g3" -> TInt32(), "g4" -> TFloat64()))
    val tab1 = TableLiteral(TableValue(t1, BroadcastRow(Row(1, 1.1), t1.globalType, sc), RVD.empty(sc, t1.canonicalRVDType)))
    val tab2 = TableLiteral(TableValue(t2, BroadcastRow(Row(2, 2.2), t2.globalType, sc), RVD.empty(sc, t2.canonicalRVDType)))

    assertEvalsTo(TableGetGlobals(TableJoin(tab1, tab2, "left")), Row(1, 1.1, 2, 2.2))
    assertEvalsTo(TableGetGlobals(TableMapGlobals(tab1, InsertFields(Ref("global", t1.globalType), Seq("g1" -> I32(3))))), Row(3, 1.1))
    assertEvalsTo(TableGetGlobals(TableRename(tab1, Map.empty, Map("g2" -> "g3"))), Row(1, 1.1))
  }



  @Test def testAggLet() {
    implicit val execStrats = ExecStrategy.interpretOnly
    val ir = TableRange(2, 2)
      .aggregate(
        aggLet(a = 'row('idx).toL + I64(1)) {
          aggLet(b = 'a * I64(2)) {
            applyAggOp(Max(), seqOpArgs = FastIndexedSeq('b * 'b))
          } + aggLet(c = 'a * I64(3)) {
            applyAggOp(Sum(), seqOpArgs = FastIndexedSeq('c * 'c))
          }
        }
      )

    assertEvalsTo(ir, 61L)
  }
}

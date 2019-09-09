package is.hail.expr.ir

import is.hail.ExecStrategy.ExecStrategy
import is.hail.{ExecStrategy, HailContext, HailSuite}
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.asm4s.Code
import is.hail.expr.{Nat, ir}
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.ir.IRSuite.TestFunctions
import is.hail.expr.ir.functions._
import is.hail.expr.types.{TableType, virtual}
import is.hail.expr.types.physical.{PArray, PBoolean, PFloat32, PFloat64, PInt32, PInt64, PString, PStruct, PTuple, PType}
import is.hail.expr.types.TableType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.io.bgen.MatrixBGENReader
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.rvd.RVD
import is.hail.table.{Ascending, Descending, SortField, Table}
import is.hail.utils.{FastIndexedSeq, _}
import is.hail.variant.{Call2, Locus, MatrixTable}
import org.apache.spark.sql.Row
import org.json4s.jackson.Serialization
import org.testng.annotations.{DataProvider, Test}

import scala.language.{dynamics, implicitConversions}

object IRSuite {
  outer =>
  var globalCounter: Int = 0

  def incr(): Unit = {
    globalCounter += 1
  }

  object TestFunctions extends RegistryFunctions {

    def registerSeededWithMissingness(mname: String, aTypes: Array[Type], rType: Type, pt: Seq[PType] => PType)(impl: (EmitRegion, PType, Long, Array[(PType, EmitTriplet)]) => EmitTriplet) {
      IRFunctionRegistry.addIRFunction(new SeededIRFunction {
        val isDeterministic: Boolean = false

        override val name: String = mname

        override val argTypes: Seq[Type] = aTypes

        override val returnType: Type = rType

        override def returnPType(argTypes: Seq[PType]): PType = if (pt == null) PType.canonical(returnType) else pt(argTypes)

        def applySeeded(seed: Long, r: EmitRegion, args: (PType, EmitTriplet)*): EmitTriplet =
          impl(r, returnPType(args.map(_._1)), seed, args.toArray)
      })
    }

    def registerSeededWithMissingness(mname: String, mt1: Type, rType: Type, pt: PType => PType)(impl: (EmitRegion, PType, Long, (PType, EmitTriplet)) => EmitTriplet): Unit =
      registerSeededWithMissingness(mname, Array(mt1), rType, unwrappedApply(pt)) { case (r, rt, seed, Array(a1)) => impl(r, rt, seed, a1) }

    def registerAll() {
      registerSeededWithMissingness("incr_s", TBoolean(), TBoolean(), null) { case (mb, rt,  _, (lT, l)) =>
        EmitTriplet(Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.setup),
          l.m,
          l.v)
      }

      registerSeededWithMissingness("incr_m", TBoolean(), TBoolean(), null) { case (mb, rt, _, (lT, l)) =>
        EmitTriplet(l.setup,
          Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.m),
          l.v)
      }

      registerSeededWithMissingness("incr_v", TBoolean(), TBoolean(), null) { case (mb, rt, _, (lT, l)) =>
        EmitTriplet(l.setup,
          l.m,
          Code(Code.invokeScalaObject[Unit](outer.getClass, "incr"), l.v))
      }
    }
  }

}

class IRSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.nonLowering

  def assertPType(node: IR, expected: PType, env: Env[PType] = Env.empty) {
    InferPType(node, env)
    assert(node.pType2 == expected)
  }

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

  @Test def testScalarInferPType() {
    assertPType(I32(5), PInt32(true))
    assertPType(I64(5), PInt64(true))
    assertPType(F32(3.1415f), PFloat32(true))
    assertPType(F64(3.1415926589793238462643383), PFloat64(true))
    assertPType(Str("HELLO WORLD"), PString(true))
    assertPType(True(), PBoolean(true))
    assertPType(False(), PBoolean(true))
  }

  // FIXME Void() doesn't work because we can't handle a void type in a tuple

  @Test def testCast() {
    assertAllEvalTo(
      (Cast(I32(5), TInt32()), 5),
      (Cast(I32(5), TInt64()), 5L),
      (Cast(I32(5), TFloat32()), 5.0f),
      (Cast(I32(5), TFloat64()), 5.0),
      (Cast(I64(5), TInt32()), 5),
      (Cast(I64(0xf29fb5c9af12107dL), TInt32()), 0xaf12107d), // truncate
      (Cast(I64(5), TInt64()), 5L),
      (Cast(I64(5), TFloat32()), 5.0f),
      (Cast(I64(5), TFloat64()), 5.0),
      (Cast(F32(3.14f), TInt32()), 3),
      (Cast(F32(3.99f), TInt32()), 3), // truncate
      (Cast(F32(3.14f), TInt64()), 3L),
      (Cast(F32(3.14f), TFloat32()), 3.14f),
      (Cast(F32(3.14f), TFloat64()), 3.14),
      (Cast(F64(3.14), TInt32()), 3),
      (Cast(F64(3.99), TInt32()), 3), // truncate
      (Cast(F64(3.14), TInt64()), 3L),
      (Cast(F64(3.14), TFloat32()), 3.14f),
      (Cast(F64(3.14), TFloat64()), 3.14))
  }

  @Test def testCastInferPType() {
    assertPType(Cast(I32(5), TInt32()), PInt32(true))
    assertPType(Cast(I32(5), TInt64()), PInt64(true))
    assertPType(Cast(I32(5), TFloat32()), PFloat32(true))
    assertPType(Cast(I32(5), TFloat64()), PFloat64(true))

    assertPType(Cast(I64(5), TInt32()), PInt32(true))
    assertPType(Cast(I64(0xf29fb5c9af12107dL), TInt32()), PInt32(true)) // truncate
    assertPType(Cast(I64(5), TInt64()), PInt64(true))
    assertPType(Cast(I64(5), TFloat32()), PFloat32(true))
    assertPType(Cast(I64(5), TFloat64()), PFloat64(true))

    assertPType(Cast(F32(3.14f), TInt32()), PInt32(true))
    assertPType(Cast(F32(3.99f), TInt32()), PInt32(true)) // truncate
    assertPType(Cast(F32(3.14f), TInt64()), PInt64(true))
    assertPType(Cast(F32(3.14f), TFloat32()), PFloat32(true))
    assertPType(Cast(F32(3.14f), TFloat64()), PFloat64(true))

    assertPType(Cast(F64(3.14), TInt32()), PInt32(true))
    assertPType(Cast(F64(3.99), TInt32()), PInt32(true)) // truncate
    assertPType(Cast(F64(3.14), TInt64()), PInt64(true))
    assertPType(Cast(F64(3.14), TFloat32()), PFloat32(true))
    assertPType(Cast(F64(3.14), TFloat64()), PFloat64(true))
  }

  @Test def testCastRename() {
    assertEvalsTo(CastRename(MakeStruct(FastSeq(("x", I32(1)))), TStruct("foo" -> TInt32())), Row(1))
    assertEvalsTo(CastRename(MakeArray(FastSeq(MakeStruct(FastSeq(("x", I32(1))))),
      TArray(TStruct("x" -> TInt32()))), TArray(TStruct("foo" -> TInt32()))),
      FastIndexedSeq(Row(1)))
  }

  @Test def testNA() {
    assertEvalsTo(NA(TInt32()), null)
  }

  @Test def testNAIsNAInferPType() {
    assertPType(NA(TInt32()), PInt32(false))

    assertPType(IsNA(NA(TInt32())), PBoolean(true))
    assertPType(IsNA(I32(5)), PBoolean(true))
  }

  @Test def testCoalesce() {
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32()))), FastIndexedSeq((null, TInt32())), null)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32()))), FastIndexedSeq((1, TInt32())), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32()), In(0, TInt32()))), FastIndexedSeq((null, TInt32())), null)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32()), In(0, TInt32()))), FastIndexedSeq((1, TInt32())), 1)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32()), NA(TInt32()))), FastIndexedSeq((1, TInt32())), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32()), I32(1), I32(1), NA(TInt32()), I32(1), NA(TInt32()), I32(1))), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32()), I32(1), Die("foo", TInt32()))), 1)(ExecStrategy.javaOnly)
  }

  val i32na = NA(TInt32())
  val i64na = NA(TInt64())
  val f32na = NA(TFloat32())
  val f64na = NA(TFloat64())
  val bna = NA(TBoolean())

  @Test def testApplyUnaryPrimOpNegate() {
    assertAllEvalTo(
      (ApplyUnaryPrimOp(Negate(), I32(5)), -5),
      (ApplyUnaryPrimOp(Negate(), i32na), null),
      (ApplyUnaryPrimOp(Negate(), I64(5)), -5L),
      (ApplyUnaryPrimOp(Negate(), i64na), null),
      (ApplyUnaryPrimOp(Negate(), F32(5)), -5F),
      (ApplyUnaryPrimOp(Negate(), f32na), null),
      (ApplyUnaryPrimOp(Negate(), F64(5)), -5D),
      (ApplyUnaryPrimOp(Negate(), f64na), null)
    )
  }

  @Test def testApplyUnaryPrimOpBang() {
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), False()), true)
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), True()), false)
    assertEvalsTo(ApplyUnaryPrimOp(Bang(), bna), null)
  }

  @Test def testApplyUnaryPrimOpBitFlip() {
    assertAllEvalTo(
      (ApplyUnaryPrimOp(BitNot(), I32(0xdeadbeef)), ~0xdeadbeef),
      (ApplyUnaryPrimOp(BitNot(), I32(-0xdeadbeef)), ~(-0xdeadbeef)),
      (ApplyUnaryPrimOp(BitNot(), i32na), null),
      (ApplyUnaryPrimOp(BitNot(), I64(0xdeadbeef12345678L)), ~0xdeadbeef12345678L),
      (ApplyUnaryPrimOp(BitNot(), I64(-0xdeadbeef12345678L)), ~(-0xdeadbeef12345678L)),
      (ApplyUnaryPrimOp(BitNot(), i64na), null)
    )
  }

  @Test def testApplyUnaryPrimOpInferPType() {
    val i32na = NA(TInt32())
    def i64na = NA(TInt64())
    def f32na = NA(TFloat32())
    def f64na = NA(TFloat64())
    def bna = NA(TBoolean())

    var node = ApplyUnaryPrimOp(Negate(), I32(5))
    assertPType(node, PInt32(true))
    node = ApplyUnaryPrimOp(Negate(), i32na)
    assertPType(node, PInt32(false))

    // should not be able to infer physical type twice on one IR (i32na)
    node = ApplyUnaryPrimOp(Negate(), i32na)
    intercept[AssertionError](InferPType(node, Env.empty))

    node = ApplyUnaryPrimOp(Negate(), I64(5))
    assertPType(node, PInt64(true))

    node = ApplyUnaryPrimOp(Negate(), i64na)
    assertPType(node, PInt64(false))

    node = ApplyUnaryPrimOp(Negate(), F32(5))
    assertPType(node, PFloat32(true))

    node = ApplyUnaryPrimOp(Negate(), f32na)
    assertPType(node, PFloat32(false))

    node = ApplyUnaryPrimOp(Negate(), F64(5))
    assertPType(node, PFloat64(true))

    node = ApplyUnaryPrimOp(Negate(), f64na)
    assertPType(node, PFloat64(false))

    node = ApplyUnaryPrimOp(Bang(), False())
    assertPType(node, PBoolean(true))

    node = ApplyUnaryPrimOp(Bang(), True())
    assertPType(node, PBoolean(true))

    node = ApplyUnaryPrimOp(Bang(), bna)
    assertPType(node, PBoolean(false))

    node = ApplyUnaryPrimOp(BitNot(), I32(0xdeadbeef))
    assertPType(node, PInt32(true))

    node = ApplyUnaryPrimOp(BitNot(), I64(0xdeadbeef12345678L))
    assertPType(node, PInt64(true))

    node = ApplyUnaryPrimOp(BitNot(), I64(-0xdeadbeef12345678L))
    assertPType(node, PInt64(true))

    node = ApplyUnaryPrimOp(BitNot(), i64na)
    assertPType(node, PInt64(false))
  }

  @Test def testComplexInferPType() {
    var ir = ArrayMap(
      Let(
        "q",
        I32(2),
        ArrayMap(
          Let(
            "v",
            Ref("q", TInt32()) + I32(3),
            ArrayRange(0, Ref("v", TInt32()), 1)
          ),
          "x",
          Ref("x", TInt32()) + Ref("q", TInt32())
        )
      ),
      "y",
      Ref("y", TInt32()) + I32(3))

    assertPType(ir, PArray(PInt32(true), true))
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

  @Test def testMakeStructInferPType() {
    var ir = MakeStruct(FastSeq())
    assertPType(ir, PStruct(true))

    ir = MakeStruct(FastSeq("a" -> NA(TInt32()), "b" -> 4, "c" -> 0.5))
    assertPType(ir, PStruct(true, "a" -> PInt32(false), "b" -> PInt32(true), "c" -> PFloat64(true)))

    val ir2 = GetField(MakeStruct((0 until 20000).map(i => s"foo$i" -> I32(1))), "foo1")
    assertPType(ir2, PInt32(true))
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
    assertEvalsTo(MakeTuple.ordered(FastSeq()), Row())
    assertEvalsTo(MakeTuple.ordered(FastSeq(NA(TInt32()), 4, 0.5)), Row(null, 4, 0.5))
    //making sure wide structs get emitted without failure
    assertEvalsTo(GetTupleElement(MakeTuple.ordered((0 until 20000).map(I32)), 1), 1)
  }

  @Test def testGetTupleElement() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = MakeTuple.ordered(FastIndexedSeq(I32(5), Str("abc"), NA(TInt32())))
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

    assertEvalsTo(ToDict(NA(TArray(TTuple(FastIndexedSeq(TInt32(), TString()): _*)))), null)

    val a = MakeArray(FastIndexedSeq(
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("a"))),
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("a"))), // duplicate key-value pair
      MakeTuple.ordered(FastIndexedSeq(NA(TInt32()), Str("b"))),
      MakeTuple.ordered(FastIndexedSeq(I32(3), NA(TString()))),
      NA(TTuple(FastIndexedSeq(TInt32(), TString()): _*)) // missing value
    ), TArray(TTuple(FastIndexedSeq(TInt32(), TString()): _*)))

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
    assertEvalsTo(invoke("contains", TBoolean(), NA(t), I32(2)), null)

    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), NA(TInt32())),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(2)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(0)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      false)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(7)),
      FastIndexedSeq((Set(-7, 2), t)),
      false)
  }

  @Test def testDictContains() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TDict(TInt32(), TString())
    assertEvalsTo(invoke("contains", TBoolean(), NA(t), I32(2)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), NA(TInt32())),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(2)),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(0)),
      FastIndexedSeq((d, t)),
      false)
    assertEvalsTo(invoke("contains", TBoolean(), In(0, t), I32(3)),
      FastIndexedSeq((Map(1 -> "a", 2 -> null), t)),
      false)
  }

  @Test def testLowerBoundOnOrderedCollectionArray() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TArray(TInt32()))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val awoutna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4)), TArray(TInt32()))
    val awna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4), NA(TInt32())), TArray(TInt32()))
    val awdups = MakeArray(FastIndexedSeq(I32(0), I32(0), I32(2), I32(4), I32(4), NA(TInt32())), TArray(TInt32()))
    assertAllEvalTo(
      (LowerBoundOnOrderedCollection(awoutna, I32(-1), onKey = false), 0),
        (LowerBoundOnOrderedCollection(awoutna, I32(0), onKey = false), 0),
        (LowerBoundOnOrderedCollection(awoutna, I32(1), onKey = false), 1),
        (LowerBoundOnOrderedCollection(awoutna, I32(2), onKey = false), 1),
        (LowerBoundOnOrderedCollection(awoutna, I32(3), onKey = false), 2),
        (LowerBoundOnOrderedCollection(awoutna, I32(4), onKey = false), 2),
        (LowerBoundOnOrderedCollection(awoutna, I32(5), onKey = false), 3),
        (LowerBoundOnOrderedCollection(awoutna, NA(TInt32()), onKey = false), 3),
        (LowerBoundOnOrderedCollection(awna, NA(TInt32()), onKey = false), 3),
        (LowerBoundOnOrderedCollection(awna, I32(5), onKey = false), 3),
        (LowerBoundOnOrderedCollection(awdups, I32(0), onKey = false), 0),
        (LowerBoundOnOrderedCollection(awdups, I32(4), onKey = false), 3)
    )
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

    val b = MakeArray(FastIndexedSeq(
      MakeArray(FastIndexedSeq(I32(7), I32(0)), ta),
      NA(ta),
      MakeArray(FastIndexedSeq(I32(2)), ta)),
      taa)
    assertEvalsTo(Let("a", I32(5), ArrayFlatMap(b, "b", Ref("b", ta))), FastIndexedSeq(7, 0, 2))

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
    assertEvalsTo(fold(TestUtils.IRArray(1, null, 3), NA(TInt32()), (accum, elt) => I32(5) + I32(5)), 10)
  }

  @Test def testArrayScan() {
    implicit val execStrats = ExecStrategy.javaOnly

    def scan(array: IR, zero: IR, f: (IR, IR) => IR): IR =
      ArrayScan(array, zero, "_accum", "_elt", f(Ref("_accum", zero.typ), Ref("_elt", zero.typ)))

    assertEvalsTo(scan(ArrayRange(1, 4, 1), NA(TBoolean()), (accum, elt) => IsNA(accum)), FastIndexedSeq(null, true, false, false))
    assertEvalsTo(scan(TestUtils.IRArray(1, 2, 3), 0, (accum, elt) => accum + elt), FastIndexedSeq(0, 1, 3, 6))
    assertEvalsTo(scan(TestUtils.IRArray(1, 2, 3), NA(TInt32()), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(TestUtils.IRArray(1, null, 3), NA(TInt32()), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(NA(TArray(TInt32())), 0, (accum, elt) => accum + elt), null)
  }

  def makeNDArray(data: Seq[Double], shape: Seq[Long], rowMajor: IR): MakeNDArray = {
    MakeNDArray(MakeArray(data.map(F64), TArray(TFloat64())), MakeTuple.ordered(shape.map(I64)), rowMajor)
  }

  def makeNDArrayRef(nd: IR, indxs: IndexedSeq[Long]): NDArrayRef = NDArrayRef(nd, indxs.map(I64))

  val scalarRowMajor = makeNDArray(FastSeq(3.0), FastSeq(), True())
  val scalarColMajor = makeNDArray(FastSeq(3.0), FastSeq(), False())
  val vectorRowMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), True())
  val vectorColMajor = makeNDArray(FastSeq(1.0, -1.0), FastSeq(2), False())
  val matrixRowMajor = makeNDArray(FastSeq(1.0, 2.0, 3.0, 4.0), FastSeq(2, 2), True())
  val threeTensorRowMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), True())
  val threeTensorColMajor = makeNDArray((0 until 30).map(_.toDouble), FastSeq(2, 3, 5), False())
  val cubeRowMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), True())
  val cubeColMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), False())

  @Test def testNDArrayShape() {
    implicit val execStrats = Set(ExecStrategy.JvmCompile)

    assertEvalsTo(NDArrayShape(scalarRowMajor), Row())
    assertEvalsTo(NDArrayShape(vectorRowMajor), Row(2L))
    assertEvalsTo(NDArrayShape(cubeRowMajor), Row(3L, 3L, 3L))
  }

  @Test def testNDArrayRef() {
    implicit val execStrats: Set[ExecStrategy] = Set()

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
    assertEvalsTo(sevenRowMajor, 7.0)
    assertEvalsTo(sevenColMajor, 7.0)

    val cubeRowMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), True())
    val cubeColMajor = makeNDArray((0 until 27).map(_.toDouble), FastSeq(3, 3, 3), False())
    val centerRowMajor = makeNDArrayRef(cubeRowMajor, FastSeq(1, 1, 1))
    val centerColMajor = makeNDArrayRef(cubeColMajor, FastSeq(1, 1, 1))
    assertEvalsTo(centerRowMajor, 13.0)
    assertEvalsTo(centerColMajor, 13.0)
  }

  @Test def testNDArrayReshape() {
    implicit val execStrats: Set[ExecStrategy] = Set()
    val v = NDArrayReshape(matrixRowMajor, MakeTuple.ordered(Seq(I64(4))))
    val mat2 = NDArrayReshape(v, MakeTuple.ordered(Seq(I64(2), I64(2))))

    assertEvalsTo(makeNDArrayRef(v, FastIndexedSeq(2)), 3.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastIndexedSeq(1, 0)), 3.0)
    assertEvalsTo(makeNDArrayRef(v, FastIndexedSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastIndexedSeq(0, 0)), 1.0)
  }

  @Test def testNDArrayMap() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val data = 0 until 10
    val shape = FastSeq(2L, 5L)
    val nDim = 2

    val positives = makeNDArray(data.map(_.toDouble), shape, True())
    val negatives = NDArrayMap(positives, "e", ApplyUnaryPrimOp(Negate(), Ref("e", TFloat64())))
    assertEvalsTo(makeNDArrayRef(positives, FastSeq(1L, 0L)), 5.0)
    assertEvalsTo(makeNDArrayRef(negatives, FastSeq(1L, 0L)), -5.0)

    val trues = MakeNDArray(MakeArray(data.map(_ => True()), TArray(TBoolean())), MakeTuple.ordered(shape.map(I64)), True())
    val falses = NDArrayMap(trues, "e", ApplyUnaryPrimOp(Bang(), Ref("e", TBoolean())))
    assertEvalsTo(makeNDArrayRef(trues, FastSeq(1L, 0L)), true)
    assertEvalsTo(makeNDArrayRef(falses, FastSeq(1L, 0L)), false)

    val bools = MakeNDArray(MakeArray(data.map(i => if (i % 2 == 0) True() else False()), TArray(TBoolean())),
      MakeTuple.ordered(shape.map(I64)), False())
    val boolsToBinary = NDArrayMap(bools, "e", If(Ref("e", TBoolean()), I64(1L), I64(0L)))
    val one = makeNDArrayRef(boolsToBinary, FastSeq(0L, 0L))
    val zero = makeNDArrayRef(boolsToBinary, FastSeq(1L, 1L))
    assertEvalsTo(one, 1L)
    assertEvalsTo(zero, 0L)
  }

  @Test def testNDArrayMap2() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val shape = MakeTuple.ordered(FastSeq(2L, 2L).map(I64))
    val numbers = MakeNDArray(MakeArray((0 until 4).map { i => F64(i.toDouble) }, TArray(TFloat64())), shape, True())
    val bools = MakeNDArray(MakeArray(Seq(True(), False(), False(), True()), TArray(TBoolean())), shape, True())

    val actual = NDArrayMap2(numbers, bools, "n", "b",
      ApplyBinaryPrimOp(Add(), Ref("n", TFloat64()), If(Ref("b", TBoolean()), F64(10), F64(20))))
    val ten = makeNDArrayRef(actual, FastSeq(0L, 0L))
    val twentyTwo = makeNDArrayRef(actual, FastSeq(1L, 0L))
    assertEvalsTo(ten, 10.0)
    assertEvalsTo(twentyTwo, 22.0)
  }

  @Test def testNDArrayReindex() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val transpose = NDArrayReindex(matrixRowMajor, FastIndexedSeq(1, 0))
    val identity = NDArrayReindex(matrixRowMajor, FastIndexedSeq(0, 1))

    val topLeftIndex = FastSeq(0L, 0L)
    val bottomLeftIndex = FastSeq(1L, 0L)

    assertEvalsTo(makeNDArrayRef(matrixRowMajor, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(identity, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(transpose, topLeftIndex), 1.0)
    assertEvalsTo(makeNDArrayRef(matrixRowMajor, bottomLeftIndex), 3.0)
    assertEvalsTo(makeNDArrayRef(identity, bottomLeftIndex), 3.0)
    assertEvalsTo(makeNDArrayRef(transpose, bottomLeftIndex), 2.0)

    val partialTranspose = NDArrayReindex(cubeRowMajor, FastIndexedSeq(0, 2, 1))
    val idx = FastIndexedSeq(0L, 1L, 0L)
    val partialTranposeIdx = FastIndexedSeq(0L, 0L, 1L)
    assertEvalsTo(makeNDArrayRef(cubeRowMajor, idx), 3.0)
    assertEvalsTo(makeNDArrayRef(partialTranspose, partialTranposeIdx), 3.0)
  }

  @Test def testNDArrayBroadcasting() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val scalarWithMatrix = NDArrayMap2(
      NDArrayReindex(scalarRowMajor, FastIndexedSeq(1, 0)),
      matrixRowMajor,
      "s", "m",
      ApplyBinaryPrimOp(Add(), Ref("s", TFloat64()), Ref("m", TFloat64())))

    val topLeft = makeNDArrayRef(scalarWithMatrix, FastIndexedSeq(0, 0))
    assertEvalsTo(topLeft, 4.0)

    val vectorWithMatrix = NDArrayMap2(
      NDArrayReindex(vectorRowMajor, FastIndexedSeq(1, 0)),
      matrixRowMajor,
      "v", "m",
      ApplyBinaryPrimOp(Add(), Ref("v", TFloat64()), Ref("m", TFloat64())))

    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(0, 1)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(1, 0)), 4.0)

    val colVector = makeNDArray(FastIndexedSeq(1.0, -1.0), FastIndexedSeq(2, 1), True())
    val colVectorWithMatrix = NDArrayMap2(colVector, matrixRowMajor, "v", "m",
      ApplyBinaryPrimOp(Add(), Ref("v", TFloat64()), Ref("m", TFloat64())))

    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(0, 1)), 3.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(1, 0)), 2.0)
  }

  @Test def testNDArrayAgg() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val three = makeNDArrayRef(NDArrayAgg(scalarRowMajor, IndexedSeq.empty), IndexedSeq.empty)
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

  @Test def testNDArrayMatMul() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val dotProduct = NDArrayMatMul(vectorRowMajor, vectorRowMajor)
    val zero = makeNDArrayRef(dotProduct, IndexedSeq())
    assertEvalsTo(zero, 2.0)

    val seven = makeNDArrayRef(NDArrayMatMul(matrixRowMajor, matrixRowMajor), IndexedSeq(0, 0))
    assertEvalsTo(seven, 7.0)

    val twoByThreeByFive = threeTensorRowMajor
    val twoByFiveByThree = NDArrayReindex(twoByThreeByFive, IndexedSeq(0, 2, 1))
    val twoByThreeByThree = NDArrayMatMul(twoByThreeByFive, twoByFiveByThree)
    val thirty = makeNDArrayRef(twoByThreeByThree, IndexedSeq(0, 0, 0))
    assertEvalsTo(thirty, 30.0)

    val threeByTwoByFive = NDArrayReindex(twoByThreeByFive, IndexedSeq(1, 0, 2))
    val matMulCube = NDArrayMatMul(NDArrayReindex(matrixRowMajor, IndexedSeq(2, 0, 1)), threeByTwoByFive)
    assertEvalsTo(makeNDArrayRef(matMulCube, IndexedSeq(0, 0, 0)), 30.0)
  }

  @Test def testNDArraySlice() {
    implicit val execStrats: Set[ExecStrategy] = Set()

    val rightCol = NDArraySlice(matrixRowMajor, MakeTuple.ordered(Seq(MakeTuple.ordered(Seq(I64(0), I64(2), I64(1))), I64(1))))
    assertEvalsTo(NDArrayShape(rightCol), Row(2L))
    assertEvalsTo(makeNDArrayRef(rightCol, FastIndexedSeq(0)), 2.0)
    assertEvalsTo(makeNDArrayRef(rightCol, FastIndexedSeq(1)), 4.0)

    val topRow = NDArraySlice(matrixRowMajor,
      MakeTuple.ordered(Seq(I64(0),
      MakeTuple.ordered(Seq(I64(0), GetTupleElement(NDArrayShape(matrixRowMajor), 1), I64(1))))))
    assertEvalsTo(makeNDArrayRef(topRow, FastIndexedSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(topRow, FastIndexedSeq(1)), 2.0)

    val scalarSlice = NDArraySlice(scalarRowMajor, MakeTuple.ordered(FastSeq()))
    assertEvalsTo(makeNDArrayRef(scalarSlice, FastIndexedSeq()), 3.0)
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

  @Test def testArrayAggContexts() {
    implicit val execStrats = Set(ExecStrategy.JvmCompile)

    val ir = Let(
      "x",
      In(0, TInt32()) * In(0, TInt32()), // multiply to prevent forwarding
      ArrayAgg(
        ArrayRange(I32(0), I32(10), I32(1)),
        "elt",
        AggLet("y",
          Cast(Ref("x", TInt32()) * Ref("x", TInt32()) * Ref("elt", TInt32()), TInt64()), // different type to trigger validation errors
          invoke("append", TArray(TArray(TInt32())),
            ApplyAggOp(FastIndexedSeq(), None, FastIndexedSeq(
              MakeArray(FastSeq(
                Ref("x", TInt32()),
                Ref("elt", TInt32()),
                Cast(Ref("y", TInt64()), TInt32()),
                Cast(Ref("y", TInt64()), TInt32())), // reference y twice to prevent forwarding
                TArray(TInt32()))),
              AggSignature(Collect(), FastIndexedSeq(), None, FastIndexedSeq(TArray(TInt32())))),
            MakeArray(FastSeq(Ref("x", TInt32())), TArray(TInt32()))),
          isScan = false
        )
      )
    )

    assertEvalsTo(ir, FastIndexedSeq(1 -> TInt32()),
      (0 until 10).map(i => FastIndexedSeq(1, i, i, i)) ++ FastIndexedSeq(FastIndexedSeq(1)))
  }

  @Test def testArrayAggScan() {
    implicit val execStrats = Set(ExecStrategy.JvmCompile)

    val eltType = TStruct("x" -> TCall(), "y" -> TInt32())

    val ir = ArrayAggScan(In(0, TArray(eltType)),
      "foo",
      GetField(Ref("foo", eltType), "y") +
        GetField(ApplyScanOp(
          FastIndexedSeq(),
          Some(FastIndexedSeq(I32(2))),
          FastIndexedSeq(GetField(Ref("foo", eltType), "x")),
          AggSignature(CallStats(), FastIndexedSeq(), Some(FastIndexedSeq(TInt32())), FastIndexedSeq(TCall()))
        ), "AN"))

    assertEvalsTo(ir,
      args = FastIndexedSeq(
        FastIndexedSeq(
          Row(null, 1),
          Row(Call2(0, 0), 2),
          Row(Call2(0, 1), 3),
          Row(Call2(1, 1), 4),
          null,
          Row(null, 5)) -> TArray(eltType)),
      expected = FastIndexedSeq(1 + 0, 2 + 0, 3 + 2, 4 + 4, null, 5 + 6))
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
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.JvmCompile)
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
    assertEvalsTo(MakeTuple.ordered(types.zip(values).map { case (t, v) => Literal(t, v) }), Row.fromSeq(values.toFastSeq))
    assertEvalsTo(Str("hello"+poopEmoji), "hello"+poopEmoji)
  }

  @Test def testSameLiteralsWithDifferentTypes() {
    assertEvalsTo(ApplyComparisonOp(EQ(TArray(TInt32())),
      ArrayMap(Literal(TArray(TFloat64()), FastIndexedSeq(1.0, 2.0)), "elt", Cast(Ref("elt", TFloat64()), TInt32())),
      Literal(TArray(TInt32()), FastIndexedSeq(1, 2))), true)
  }

  @Test def testTableCount() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
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

    def tuple(k: String, v: Int): IR = MakeTuple.ordered(Seq(Str(k), I32(v)))

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

    assertEvalsTo(ApplyComparisonOp(EQ(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(LT(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(GT(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(LTEQ(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(GTEQ(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQ(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(EQWithNA(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), true)
    assertEvalsTo(ApplyComparisonOp(NEQWithNA(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), false)
    assertEvalsTo(ApplyComparisonOp(Compare(t1, t2), In(0, t1), In(1, t2)), FastIndexedSeq(a -> t1, a -> t2), 0)
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
    val v = Ref("v", TInt32())
    val s = Ref("s", TStruct("x" -> TInt32(), "y" -> TInt64(), "z" -> TFloat64()))
    val t = Ref("t", TTuple(TInt32(), TInt64(), TFloat64()))
    val l = Ref("l", TInt32())
    val r = Ref("r", TInt32())

    val call = Ref("call", TCall())

    val collectSig = AggSignature(Collect(), Seq(), None, Seq(TInt32()))

    val sumSig = AggSignature(Sum(), Seq(), None, Seq(TInt32()))

    val callStatsSig = AggSignature(CallStats(), Seq(), Some(Seq(TInt32())), Seq(TCall()))

    val callStatsSig2 = AggSignature2(CallStats(), Seq(TInt32()), Seq(TCall()), None)
    val collectSig2 = AggSignature2(CallStats(), Seq(), Seq(TInt32()), None)

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
    val blockMatrixWriter = BlockMatrixNativeWriter(tmpDir.createLocalTempFile(), false, false, false)
    val blockMatrixMultiWriter = BlockMatrixBinaryMultiWriter(tmpDir.createLocalTempFile(), false)
    val nd = MakeNDArray(MakeArray(FastSeq(I32(-1), I32(1)), TArray(TInt32())),
      MakeTuple.ordered(FastSeq(I64(1), I64(2))),
      True())


    val irs = Array(
      i, I64(5), F32(3.14f), F64(3.14), str, True(), False(), Void(),
      Cast(i, TFloat64()),
      CastRename(NA(TStruct("a" -> TInt32())), TStruct("b" -> TInt32())),
      NA(TInt32()), IsNA(i),
      If(b, i, j),
      Coalesce(FastSeq(In(0, TInt32()), I32(1))),
      Let("v", i, v),
      AggLet("v", i, v, false),
      Ref("x", TInt32()),
      ApplyBinaryPrimOp(Add(), i, j),
      ApplyUnaryPrimOp(Negate(), i),
      ApplyComparisonOp(EQ(TInt32()), i, j),
      MakeArray(FastSeq(i, NA(TInt32()), I32(-3)), TArray(TInt32())),
      MakeStream(FastSeq(i, NA(TInt32()), I32(-3)), TStream(TInt32())),
      nd,
      NDArrayReshape(nd, MakeTuple.ordered(Seq(I64(4)))),
      NDArrayRef(nd, FastSeq(I64(1), I64(2))),
      NDArrayMap(nd, "v", ApplyUnaryPrimOp(Negate(), v)),
      NDArrayMap2(nd, nd, "l", "r", ApplyBinaryPrimOp(Add(), l, r)),
      NDArrayReindex(nd, FastIndexedSeq(0, 1)),
      NDArrayAgg(nd, FastIndexedSeq(0)),
      NDArrayWrite(nd, Str(tmpDir.createTempFile())),
      NDArrayMatMul(nd, nd),
      NDArraySlice(nd, MakeTuple.ordered(FastSeq(MakeTuple.ordered(FastSeq(F64(0), F64(2), F64(1))),
                                         MakeTuple.ordered(FastSeq(F64(0), F64(2), F64(1)))))),
      ArrayRef(a, i),
      ArrayLen(a),
      ArrayRange(I32(0), I32(5), I32(1)),
      StreamRange(I32(0), I32(5), I32(1)),
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
      ArrayAggScan(a, "x", ApplyScanOp(FastIndexedSeq.empty, None, FastIndexedSeq(Ref("x", TInt32())), sumSig)),
      AggFilter(True(), I32(0), false),
      AggExplode(NA(TArray(TInt32())), "x", I32(0), false),
      AggGroupBy(True(), I32(0), false),
      ApplyAggOp(FastIndexedSeq.empty, None, FastIndexedSeq(I32(0)), collectSig),
      ApplyAggOp(FastIndexedSeq.empty, Some(FastIndexedSeq(I32(2))), FastIndexedSeq(call), callStatsSig),
      ApplyAggOp(FastIndexedSeq(I32(10)), None, FastIndexedSeq(F64(-2.11), I32(4)), takeBySig),
      InitOp(I32(0), FastIndexedSeq(I32(2)), callStatsSig),
      SeqOp(I32(0), FastIndexedSeq(i), collectSig),
      SeqOp(I32(0), FastIndexedSeq(F64(-2.11), I32(17)), takeBySig),
      InitOp2(0, FastIndexedSeq(I32(2)), callStatsSig2),
      SeqOp2(0, FastIndexedSeq(i), collectSig2),
      CombOp2(0, 1, collectSig2),
      ResultOp2(0, FastSeq(collectSig2)),
      SerializeAggs(0, 0, CodecSpec.defaultBufferSpec, FastSeq(collectSig2)),
      DeserializeAggs(0, 0, CodecSpec.defaultBufferSpec, FastSeq(collectSig2)),
      Begin(FastIndexedSeq(Void())),
      MakeStruct(FastIndexedSeq("x" -> i)),
      SelectFields(s, FastIndexedSeq("x", "z")),
      InsertFields(s, FastIndexedSeq("x" -> i)),
      InsertFields(s, FastIndexedSeq("* x *" -> i)), // Won't parse as a simple identifier
      GetField(s, "x"),
      MakeTuple(FastIndexedSeq(2 -> i, 4 -> b)),
      GetTupleElement(t, 1),
      In(2, TFloat64()),
      Die("mumblefoo", TFloat64()),
      invoke("&&", TBoolean(), b, c), // ApplySpecial
      invoke("toFloat64", TFloat64(), i), // Apply
      Uniroot("x", F64(3.14), F64(-5.0), F64(5.0)),
      Literal(TStruct("x" -> TInt32()), Row(1)),
      TableCount(table),
      TableGetGlobals(table),
      TableCollect(table),
      TableAggregate(table, MakeStruct(Seq("foo" -> count))),
      TableToValueApply(table, ForceCountTable()),
      MatrixToValueApply(mt, ForceCountMatrixTable()),
      TableWrite(table, TableNativeWriter(tmpDir.createLocalTempFile(extension = "ht"))),
      MatrixWrite(mt, MatrixNativeWriter(tmpDir.createLocalTempFile(extension = "mt"))),
      MatrixWrite(vcf, MatrixVCFWriter(tmpDir.createLocalTempFile(extension = "vcf"))),
      MatrixWrite(vcf, MatrixPLINKWriter(tmpDir.createLocalTempFile())),
      MatrixWrite(bgen, MatrixGENWriter(tmpDir.createLocalTempFile())),
      MatrixMultiWrite(Array(mt, mt), MatrixNativeMultiWriter(tmpDir.createLocalTempFile())),
      TableMultiWrite(Array(table, table), WrappedMatrixNativeMultiWriter(MatrixNativeMultiWriter(tmpDir.createLocalTempFile()), FastIndexedSeq("foo"))),
      MatrixAggregate(mt, MakeStruct(Seq("foo" -> count))),
      BlockMatrixWrite(blockMatrix, blockMatrixWriter),
      BlockMatrixMultiWrite(IndexedSeq(blockMatrix, blockMatrix), blockMatrixMultiWriter),
      CollectDistributedArray(ArrayRange(0, 3, 1), 1, "x", "y", Ref("x", TInt32())),
      ReadPartition(Str("foo"), CodecSpec.default.makeCodecSpec2(PStruct("foo" -> PInt32(), "bar" -> PString())), TStruct("foo" -> TInt32())),
      RelationalLet("x", I32(0), I32(0))
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
        TableMultiWayZipJoin(FastIndexedSeq(read, read), " * data * ", "globals"),
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
        TableRename(read, Map("idx" -> "idx_foo"), Map("global_f32" -> "global_foo")),
        TableFilterIntervals(read, FastIndexedSeq(Interval(IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(10), 1))), keep = false),
        RelationalLetTable("x", I32(0), read)
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
        "row_idx" -> GetField(Ref("va", read.typ.rowType), "row_idx"),
        "new_f32" -> ApplyBinaryPrimOp(Add(),
          GetField(Ref("va", read.typ.rowType), "row_f32"),
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
        MatrixColsHead(range1, 3),
        MatrixExplodeCols(read, FastIndexedSeq("col_mset")),
        CastTableToMatrix(
          CastMatrixToTable(read, " # entries", " # cols"),
          " # entries",
          " # cols",
          read.typ.colKey),
        MatrixAnnotateColsTable(read, tableRead, "uid_123"),
        MatrixAnnotateRowsTable(read, tableRead, "uid_123", product=false),
        MatrixRename(read, Map("global_i64" -> "foo"), Map("col_i64" -> "bar"), Map("row_i64" -> "baz"), Map("entry_i64" -> "quam")),
        MatrixFilterIntervals(read, FastIndexedSeq(Interval(IntervalEndpoint(Row(0), -1), IntervalEndpoint(Row(10), 1))), keep = false),
        RelationalLetMatrixTable("x", I32(0), read)
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
    val transpose = BlockMatrixBroadcast(read, FastIndexedSeq(1, 0), FastIndexedSeq(2, 2), 2)
    val dot = BlockMatrixDot(read, transpose)
    val slice = BlockMatrixSlice(read, FastIndexedSeq(FastIndexedSeq(0, 2, 1), FastIndexedSeq(0, 1, 1)))

    val blockMatrixIRs = Array[BlockMatrixIR](read,
      transpose,
      dot,
      RelationalLetBlockMatrix("x", I32(0), read),
      slice)

    blockMatrixIRs.map(ir => Array(ir))
  }

  @Test(dataProvider = "valueIRs")
  def testValueIRParser(x: IR) {
    val env = IRParserEnvironment(refMap = Map(
      "c" -> TBoolean(),
      "a" -> TArray(TInt32()),
      "aa" -> TArray(TArray(TInt32())),
      "da" -> TArray(TTuple(TInt32(), TString())),
      "nd" -> TNDArray(TFloat64(), Nat(1)),
      "nd2" -> TNDArray(TArray(TString()), Nat(1)),
      "v" -> TInt32(),
      "l" -> TInt32(),
      "r" -> TInt32(),
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
      val args = FastIndexedSeq((i, TBoolean()))

      IRSuite.globalCounter = 0
      Interpret[Any](ctx, x, env, args, None, optimize = false)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      Interpret[Any](ctx, x, env, args, None)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      eval(x, env, args, None)
      assert(IRSuite.globalCounter == expectedEvaluations)
    }

    def i = In(0, TBoolean())

    def st = ApplySeeded("incr_s", FastSeq(True()), 0L, TBoolean())

    def sf = ApplySeeded("incr_s", FastSeq(True()), 0L, TBoolean())

    def sm = ApplySeeded("incr_s", FastSeq(NA(TBoolean())), 0L, TBoolean())

    def mt = ApplySeeded("incr_m", FastSeq(True()), 0L, TBoolean())

    def mf = ApplySeeded("incr_m", FastSeq(True()), 0L, TBoolean())

    def mm = ApplySeeded("incr_m", FastSeq(NA(TBoolean())), 0L, TBoolean())

    def vt = ApplySeeded("incr_v", FastSeq(True()), 0L, TBoolean())

    def vf = ApplySeeded("incr_v", FastSeq(True()), 0L, TBoolean())

    def vm = ApplySeeded("incr_v", FastSeq(NA(TBoolean())), 0L, TBoolean())

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
      .bind("flag" -> ((true, TBoolean())))
      .bind("array" -> ((FastIndexedSeq(0), TArray(TInt32()))))

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

    Interpret(ir.IRParser.parse_table_ir(irStr), ctx, optimize = false).rvd.count()
  }

  @Test def testTableGetGlobalsSimplifyRules() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t1 = TableType(TStruct("a" -> TInt32()), FastIndexedSeq("a"), TStruct("g1" -> TInt32(), "g2" -> TFloat64()))
    val t2 = TableType(TStruct("a" -> TInt32()), FastIndexedSeq("a"), TStruct("g3" -> TInt32(), "g4" -> TFloat64()))
    val tab1 = TableLiteral(TableValue(t1, BroadcastRow(ctx, Row(1, 1.1), t1.globalType), RVD.empty(sc, t1.canonicalRVDType)), ctx)
    val tab2 = TableLiteral(TableValue(t2, BroadcastRow(ctx, Row(2, 2.2), t2.globalType), RVD.empty(sc, t2.canonicalRVDType)), ctx)

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

  @Test def testRelationalLet() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val ir = RelationalLet("x", NA(TInt32()), RelationalRef("x", TInt32()))
    assertEvalsTo(ir, null)
  }


  @Test def testRelationalLetTable() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32()))
    val ir = TableAggregate(RelationalLetTable("x",
      Literal(t, FastIndexedSeq(Row(1))),
      TableParallelize(MakeStruct(FastSeq("rows" -> RelationalRef("x", t), "global" -> MakeStruct(FastSeq()))))),
      ApplyAggOp(FastIndexedSeq(), None, FastIndexedSeq(), AggSignature(Count(), FastIndexedSeq(), None, FastIndexedSeq())))
    assertEvalsTo(ir, 1L)
  }

  @Test def testRelationalLetMatrixTable() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32()))
    val m = CastTableToMatrix(
      TableMapGlobals(
        TableMapRows(
          TableRange(1, 1), InsertFields(Ref("row", TStruct("idx" -> TInt32())), FastSeq("entries" -> RelationalRef("x", t)))),
        MakeStruct(FastSeq("cols" -> MakeArray(FastSeq(MakeStruct(FastSeq("s" -> I32(0)))), TArray(TStruct("s" -> TInt32())))))),
      "entries",
      "cols",
      FastIndexedSeq())
    val ir = MatrixAggregate(RelationalLetMatrixTable("x",
      Literal(t, FastIndexedSeq(Row(1))),
      m),
      ApplyAggOp(FastIndexedSeq(), None, FastIndexedSeq(), AggSignature(Count(), FastIndexedSeq(), None, FastIndexedSeq())))
    assertEvalsTo(ir, 1L)
  }


  @DataProvider(name = "relationalFunctions")
  def relationalFunctionsData(): Array[Array[Any]] = Array(
    Array(TableFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(VEP("foo", false, 1)),
    Array(WrappedMatrixToMatrixFunction(MatrixFilterPartitions(Array(1, 2, 3), false), "foo", "baz", FastIndexedSeq("ck"))),
    Array(WrappedMatrixToTableFunction(LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b")), "foo", "bar", FastIndexedSeq("ck"))),
    Array(LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b"))),
    Array(LinearRegressionRowsChained(FastIndexedSeq(FastIndexedSeq("foo")), "bar", Array("baz"), 1, Array("a", "b"))),
    Array(LogisticRegression("firth", Array("a", "b"), "c", Array("d", "e"), Array("f", "g"))),
    Array(PoissonRegression("firth", "a", "c", Array("d", "e"), Array("f", "g"))),
    Array(Skat("a", "b", "c", "d", Array("e", "f"), false, 1, 0.1, 100)),
    Array(LocalLDPrune("x", 0.95, 123, 456)),
    Array(PCA("x", 1, false)),
    Array(PCRelate(0.00, 4096, Some(0.1), PCRelate.PhiK2K0K1)),
    Array(WindowByLocus(1)),
    Array(MatrixFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(ForceCountTable()),
    Array(ForceCountMatrixTable()),
    Array(NPartitionsTable()),
    Array(NPartitionsMatrixTable()),
    Array(WrappedMatrixToValueFunction(NPartitionsMatrixTable(), "foo", "bar", FastIndexedSeq("a", "c"))),
    Array(MatrixWriteBlockMatrix("a", false, "b", 1)),
    Array(MatrixExportEntriesByCol(1, "asd", false, true)),
    Array(GetElement(FastSeq(1, 2)))
  )

  @Test def relationalFunctionsRun(): Unit = {
    relationalFunctionsData()
  }

  @Test(dataProvider = "relationalFunctions")
  def testRelationalFunctionsSerialize(x: Any): Unit = {
    implicit val formats = RelationalFunctions.formats

    x match {
      case x: MatrixToMatrixFunction => assert(RelationalFunctions.lookupMatrixToMatrix(Serialization.write(x)) == x)
      case x: MatrixToTableFunction => assert(RelationalFunctions.lookupMatrixToTable(Serialization.write(x)) == x)
      case x: MatrixToValueFunction => assert(RelationalFunctions.lookupMatrixToValue(Serialization.write(x)) == x)
      case x: TableToTableFunction => assert(RelationalFunctions.lookupTableToTable(Serialization.write(x)) == x)
      case x: TableToValueFunction => assert(RelationalFunctions.lookupTableToValue(Serialization.write(x)) == x)
      case x: BlockMatrixToTableFunction => assert(RelationalFunctions.lookupBlockMatrixToTable(Serialization.write(x)) == x)
      case x: BlockMatrixToValueFunction => assert(RelationalFunctions.lookupBlockMatrixToValue(Serialization.write(x)) == x)
    }
  }

  @Test def testFoldWithSetup() {
    val v = In(0, TInt32())
    val cond1 = If(v.ceq(I32(3)),
      MakeArray(FastIndexedSeq(I32(1), I32(2), I32(3)), TArray(TInt32())),
      MakeArray(FastIndexedSeq(I32(4), I32(5), I32(6)), TArray(TInt32())))
    assertEvalsTo(ArrayFold(cond1, True(), "accum", "i", Ref("i", TInt32()).ceq(v)), FastIndexedSeq(0 -> TInt32()), false)
  }

  @Test def testNonCanonicalTypeParsing(): Unit = {
    val t = TTuple(FastIndexedSeq(TupleField(1, TInt64())))
    val lit = Literal(t, Row(1L))

    assert(IRParser.parseType(t.parsableString()) == t)
    assert(IRParser.parse_value_ir(Pretty(lit)) == lit)
  }

  @Test def regressionTestUnifyBug(): Unit = {
    // failed due to misuse of Type.unify
    val ir = IRParser.parse_value_ir(
      """
        |(ArrayMap __uid_3
        |    (Literal Array[Interval[Locus(GRCh37)]] "[{\"start\": {\"contig\": \"20\", \"position\": 10277621}, \"end\": {\"contig\": \"20\", \"position\": 11898992}, \"includeStart\": true, \"includeEnd\": false}]")
        |    (Apply Interval Interval[Struct{locus:Locus(GRCh37)}]
        |       (MakeStruct (locus  (Apply start Locus(GRCh37) (Ref __uid_3))))
        |       (MakeStruct (locus  (Apply end Locus(GRCh37) (Ref __uid_3)))) (True) (False)))
        |""".stripMargin)
    val (v, _) = HailContext.backend.execute(ir, optimize = true)
    assert(
      ir.typ.ordering.equiv(
        FastIndexedSeq(
          Interval(
            Row(Locus("20", 10277621)), Row(Locus("20", 11898992)), includesStart = true, includesEnd = false)),
        v))
  }
}

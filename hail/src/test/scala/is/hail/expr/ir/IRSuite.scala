package is.hail.expr.ir

import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.asm4s.Code
import is.hail.expr.ir.ArrayZipBehavior.ArrayZipBehavior
import is.hail.expr.ir.IRBuilder._
import is.hail.expr.ir.IRSuite.TestFunctions
import is.hail.expr.ir.functions._
import is.hail.types.TableType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types.encoded._
import is.hail.expr.Nat
import is.hail.expr.ir.agg.{CallStatsStateSig, CollectStateSig, GroupedAggSig, PhysicalAggSig, TypedStateSig}
import is.hail.io.bgen.{IndexBgen, MatrixBGENReader}
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.linalg.BlockMatrix
import is.hail.methods._
import is.hail.rvd.{RVD, RVDPartitioner, RVDSpecMaker}
import is.hail.utils.{FastIndexedSeq, _}
import is.hail.variant.{Call2, Locus}
import is.hail.{ExecStrategy, HailContext, HailSuite}
import org.apache.spark.sql.Row
import org.json4s.jackson.{JsonMethods, Serialization}
import org.testng.annotations.{DataProvider, Test}

import scala.language.{dynamics, implicitConversions}

object IRSuite {
  outer =>
  var globalCounter: Int = 0

  def incr(): Unit = {
    globalCounter += 1
  }

  object TestFunctions extends RegistryFunctions {

    def registerSeededWithMissingness(
      name: String,
      valueParameterTypes: Array[Type],
      returnType: Type,
      calculateReturnType: (Type, Seq[PType]) => PType
    )(
      impl: (EmitCodeBuilder, EmitRegion, PType, Long, Array[() => IEmitCode]) => IEmitCode
    ) {
      IRFunctionRegistry.addJVMFunction(
        new SeededMissingnessAwareJVMFunction(name, valueParameterTypes, returnType, calculateReturnType) {
          val isDeterministic: Boolean = false
          def applySeededI(seed: Long, cb: EmitCodeBuilder, r: EmitRegion, returnPType: PType, args: (PType, () => IEmitCode)*): IEmitCode = {
            assert(unify(FastSeq(), args.map(_._1.virtualType), returnPType.virtualType))
            impl(cb, r, returnPType, seed, args.map(a => a._2).toArray)
          }
        }
      )
    }

    def registerSeededWithMissingness(
      name: String,
      valueParameterType: Type,
      returnType: Type,
      calculateReturnType: (Type, PType) => PType
    )(
      impl: (EmitCodeBuilder, EmitRegion, PType, Long, () => IEmitCode) => IEmitCode
    ): Unit =
      registerSeededWithMissingness(name, Array(valueParameterType), returnType, unwrappedApply(calculateReturnType)) {
        case (cb, r, rt, seed, Array(a1)) => impl(cb, r, rt, seed, a1)
      }

    def registerAll() {
      registerSeededWithMissingness("incr_s", TBoolean, TBoolean, null) { case (cb, mb, rt,  _, l) =>
        cb += Code.invokeScalaObject0[Unit](outer.getClass, "incr")
        l()
      }

      registerSeededWithMissingness("incr_v", TBoolean, TBoolean, null) { case (cb, mb, rt, _, l) =>
        l().map(cb) { pc =>
          cb += Code.invokeScalaObject0[Unit](outer.getClass, "incr")
          pc
        }
      }
    }
  }

}

class IRSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.nonLowering

  def assertPType(node: IR, expected: PType) {
    InferPType(node)
    assert(node.pType == expected)
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
    assertPType(Str("HELLO WORLD"), PCanonicalString(true))
    assertPType(True(), PBoolean(true))
    assertPType(False(), PBoolean(true))
  }

  // FIXME Void() doesn't work because we can't handle a void type in a tuple

  @Test def testCast() {
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
      (Cast(F64(3.14), TFloat64), 3.14))
  }

  @Test def testCastInferPType() {
    assertPType(Cast(I32(5), TInt32), PInt32(true))
    assertPType(Cast(I32(5), TInt64), PInt64(true))
    assertPType(Cast(I32(5), TFloat32), PFloat32(true))
    assertPType(Cast(I32(5), TFloat64), PFloat64(true))

    assertPType(Cast(I64(5), TInt32), PInt32(true))
    assertPType(Cast(I64(0xf29fb5c9af12107dL), TInt32), PInt32(true)) // truncate
    assertPType(Cast(I64(5), TInt64), PInt64(true))
    assertPType(Cast(I64(5), TFloat32), PFloat32(true))
    assertPType(Cast(I64(5), TFloat64), PFloat64(true))

    assertPType(Cast(F32(3.14f), TInt32), PInt32(true))
    assertPType(Cast(F32(3.99f), TInt32), PInt32(true)) // truncate
    assertPType(Cast(F32(3.14f), TInt64), PInt64(true))
    assertPType(Cast(F32(3.14f), TFloat32), PFloat32(true))
    assertPType(Cast(F32(3.14f), TFloat64), PFloat64(true))

    assertPType(Cast(F64(3.14), TInt32), PInt32(true))
    assertPType(Cast(F64(3.99), TInt32), PInt32(true)) // truncate
    assertPType(Cast(F64(3.14), TInt64), PInt64(true))
    assertPType(Cast(F64(3.14), TFloat32), PFloat32(true))
    assertPType(Cast(F64(3.14), TFloat64), PFloat64(true))
  }

  @Test def testCastRename() {
    assertEvalsTo(CastRename(MakeStruct(FastSeq(("x", I32(1)))), TStruct("foo" -> TInt32)), Row(1))
    assertEvalsTo(CastRename(MakeArray(FastSeq(MakeStruct(FastSeq(("x", I32(1))))),
      TArray(TStruct("x" -> TInt32))), TArray(TStruct("foo" -> TInt32))),
      FastIndexedSeq(Row(1)))
  }

  @Test def testCastRenameIR() {
    var expectedPType: PType = PCanonicalStruct(true, "foo" -> PInt32(true))
    var childPType: PType = PCanonicalStruct(true, "x" -> PInt32(true))
    var targetType: Type = TStruct("foo" -> TInt32)
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalArray(PCanonicalStruct(true, "foo" -> PInt64(true)))
    childPType = PCanonicalArray(PCanonicalStruct(true, "c" -> PInt64(true)))
    targetType = TArray(TStruct("foo" -> TInt64))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalArray(PCanonicalStruct("foo" -> PCanonicalString(true)))
    childPType = PCanonicalArray(PCanonicalStruct("q" -> PCanonicalString(true)))
    targetType = TArray(TStruct("foo" -> TString))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalArray(PCanonicalStruct(true, "foo" -> PCanonicalStruct("baz" -> PBoolean(true))))
    childPType = PCanonicalArray(PCanonicalStruct(true, "b" -> PCanonicalStruct("a" -> PBoolean(true))))
    targetType = TArray(TStruct("foo" -> TStruct("baz" -> TBoolean)))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalArray(PCanonicalStruct("foo" -> PCanonicalArray(PFloat64(true), true), "bar" -> PCanonicalBinary()))
    childPType = PCanonicalArray(PCanonicalStruct("x" -> PCanonicalArray(PFloat64(true), true), "y" -> PCanonicalBinary()))
    targetType = TArray(TStruct("foo" -> TArray(TFloat64), "bar" -> TBinary))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalTuple(true, PCanonicalStruct(true, "foo" -> PCanonicalInterval(PFloat32())), PCanonicalStruct(false, "bar" -> PFloat64(true)))
    childPType = PCanonicalTuple(true, PCanonicalStruct(true, "v" -> PCanonicalInterval(PFloat32())), PCanonicalStruct(false, "q" -> PFloat64(true)))
    targetType = TTuple(TStruct("foo" -> TInterval(TFloat32)), TStruct("bar" -> TFloat64))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalDict(PCanonicalString(), PCanonicalTuple(false,
      PCanonicalStruct("foo" -> PCanonicalStruct("bar" -> PCanonicalNDArray(PInt32(true), 3, true))),
      PCanonicalStruct(false, "bar" -> PCanonicalBinary(true))))
    childPType = PCanonicalDict(PCanonicalString(), PCanonicalTuple(false,
      PCanonicalStruct("xxxxxx" -> PCanonicalStruct("qqq" -> PCanonicalNDArray(PInt32(true), 3, true))),
      PCanonicalStruct(false, "ddd" -> PCanonicalBinary(true))))
    targetType = TDict(TString, TTuple(TStruct("foo" -> TStruct("bar" -> TNDArray(TInt32, Nat(3)))),
      TStruct("bar" -> TBinary)))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)

    expectedPType = PCanonicalStream(PCanonicalStruct("foo2a" -> PCanonicalArray(PFloat64(true), true), "bar2a" -> PCanonicalBinary()))
    childPType = PCanonicalStream(PCanonicalStruct("q" -> PCanonicalArray(PFloat64(true), true), "yxxx" -> PCanonicalBinary()))
    targetType = TStream(TStruct("foo2a" -> TArray(TFloat64), "bar2a" -> TBinary))
    assertPType(CastRename(In(0, childPType), targetType), expectedPType)
  }

  @Test def testNA() {
    assertEvalsTo(NA(TInt32), null)
  }

  @Test def testNAIsNAInferPType() {
    assertPType(NA(TInt32), PInt32(false))

    assertPType(IsNA(NA(TInt32)), PBoolean(true))
    assertPType(IsNA(I32(5)), PBoolean(true))
  }

  @Test def testCoalesce() {
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32))), FastIndexedSeq((null, TInt32)), null)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32))), FastIndexedSeq((1, TInt32)), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), In(0, TInt32))), FastIndexedSeq((null, TInt32)), null)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), In(0, TInt32))), FastIndexedSeq((1, TInt32)), 1)
    assertEvalsTo(Coalesce(FastSeq(In(0, TInt32), NA(TInt32))), FastIndexedSeq((1, TInt32)), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), I32(1), I32(1), NA(TInt32), I32(1), NA(TInt32), I32(1))), 1)
    assertEvalsTo(Coalesce(FastSeq(NA(TInt32), I32(1), Die("foo", TInt32))), 1)
  }

  @Test def testCoalesceWithDifferentRequiredeness() {
    val t1 = In(0, TArray(TInt32))
    val t2 = NA(TArray(TInt32))
    val value = FastIndexedSeq(1, 2, 3, 4)

    assertEvalsTo(Coalesce(FastSeq(t1, t2)), FastIndexedSeq((value, TArray(TInt32))), value)
  }

  @Test def testCoalesceInferPType() {
    assertPType(Coalesce(FastSeq(In(0, PInt32()))), PInt32())
    assertPType(Coalesce(FastSeq(In(0, PInt32()), In(0, PInt32(true)))), PInt32(true))
    assertPType(Coalesce(FastSeq(In(0, PCanonicalArray(PCanonicalArray(PInt32()))), In(0, PCanonicalArray(PCanonicalArray(PInt32(true)))))), PCanonicalArray(PCanonicalArray(PInt32())))
    assertPType(Coalesce(FastSeq(In(0, PCanonicalArray(PCanonicalArray(PInt32()))), In(0, PCanonicalArray(PCanonicalArray(PInt32(true), true))))), PCanonicalArray(PCanonicalArray(PInt32())))
    assertPType(Coalesce(FastSeq(In(0, PCanonicalArray(PCanonicalArray(PInt32()))), In(0, PCanonicalArray(PCanonicalArray(PInt32(true), true), true)))), PCanonicalArray(PCanonicalArray(PInt32()), true))
    assertPType(Coalesce(FastSeq(In(0, PCanonicalArray(PCanonicalArray(PInt32()))), In(0, PCanonicalArray(PCanonicalArray(PInt32(true), true), true)))), PCanonicalArray(PCanonicalArray(PInt32()), true))
    assertPType(Coalesce(FastSeq(
      In(0, PCanonicalArray(PCanonicalArray(PInt32()))),
      In(0, PCanonicalArray(PCanonicalArray(PInt32(), true))),
      In(0, PCanonicalArray(PCanonicalArray(PInt32(true)), true))
    )), PCanonicalArray(PCanonicalArray(PInt32()), true))
  }

  val i32na = NA(TInt32)
  val i64na = NA(TInt64)
  val f32na = NA(TFloat32)
  val f64na = NA(TFloat64)
  val bna = NA(TBoolean)

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
    val i32na = NA(TInt32)
    def i64na = NA(TInt64)
    def f32na = NA(TFloat32)
    def f64na = NA(TFloat64)
    def bna = NA(TBoolean)

    var node = ApplyUnaryPrimOp(Negate(), I32(5))
    assertPType(node, PInt32(true))
    node = ApplyUnaryPrimOp(Negate(), i32na)
    assertPType(node, PInt32(false))

    // should not be able to infer physical type twice on one IR (i32na)
    node = ApplyUnaryPrimOp(Negate(), i32na)
    intercept[RuntimeException](InferPType(node))

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
    // InferPType expects array->stream lowered ir
    val ir = ToArray(StreamMap(
      Let(
        "q",
        I32(2),
        StreamMap(
          Let(
            "v",
            Ref("q", TInt32) + I32(3),
            StreamRange(0, Ref("v", TInt32), 1)
          ),
          "x",
          Ref("x", TInt32) + Ref("q", TInt32)
        )
      ),
      "y",
      Ref("y", TInt32) + I32(3)))

    assertPType(ir, PCanonicalArray(PInt32(true), true))
  }

  @Test def testApplyBinaryPrimOpAdd() {
    def assertSumsTo(t: Type, x: Any, y: Any, sum: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Add(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), sum)
    }
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

  @Test def testApplyBinaryPrimOpSubtract() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Subtract(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpMultiply() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(Multiply(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpFloatingPointDivide() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(FloatingPointDivide(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

    assertExpected(TInt32, 5, 2, 2.5f)
    assertExpected(TInt32, 5, null, null)
    assertExpected(TInt32, null, 2, null)
    assertExpected(TInt32, null, null, null)

    assertExpected(TInt64, 5L, 2L, 2.5f)
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

  @Test def testApplyBinaryPrimOpRoundToNegInfDivide() {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(RoundToNegInfDivide(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpBitAnd(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitAnd(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpBitOr(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitOr(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpBitXOr(): Unit = {
    def assertExpected(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(BitXOr(), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyBinaryPrimOpLeftShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(LeftShift(), In(0, t), In(1, TInt32)), FastIndexedSeq(x -> t, y -> TInt32), expected)
    }

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

  @Test def testApplyBinaryPrimOpRightShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(RightShift(), In(0, t), In(1, TInt32)), FastIndexedSeq(x -> t, y -> TInt32), expected)
    }

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

  @Test def testApplyBinaryPrimOpLogicalRightShift(): Unit = {
    def assertShiftsTo(t: Type, x: Any, y: Any, expected: Any) {
      assertEvalsTo(ApplyBinaryPrimOp(LogicalRightShift(), In(0, t), In(1, TInt32)), FastIndexedSeq(x -> t, y -> TInt32), expected)
    }

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

  @Test def testApplyComparisonOpGT() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(GT(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyComparisonOpGTEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(GTEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyComparisonOpLT() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(LT(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyComparisonOpLTEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(LTEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyComparisonOpEQ() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(EQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testApplyComparisonOpNE() {
    def assertComparesTo(t: Type, x: Any, y: Any, expected: Boolean) {
      assertEvalsTo(ApplyComparisonOp(NEQ(t), In(0, t), In(1, t)), FastIndexedSeq(x -> t, y -> t), expected)
    }

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

  @Test def testIf() {
    assertEvalsTo(If(True(), I32(5), I32(7)), 5)
    assertEvalsTo(If(False(), I32(5), I32(7)), 7)
    assertEvalsTo(If(NA(TBoolean), I32(5), I32(7)), null)
    assertEvalsTo(If(True(), NA(TInt32), I32(7)), null)
  }

  @Test def testIfInferPType() {
    assertPType(If(True(), In(0, PInt32(true)), In(1, PInt32(true))), PInt32(true))
    assertPType(If(True(), In(0, PInt32(false)), In(1, PInt32(true))), PInt32(false))
    assertPType(If(NA(TBoolean), In(0, PInt32(true)), In(1, PInt32(true))), PInt32(false))

    var cnsqBranch = In(0, PCanonicalArray(PCanonicalArray(PInt32(true), true), true))
    var altrBranch = In(1, PCanonicalArray(PCanonicalArray(PInt32(true), true), true))

    var ir = If(True(), cnsqBranch, altrBranch)
    assertPType(ir, PCanonicalArray(PCanonicalArray(PInt32(true), true), true))

    cnsqBranch = In(0, PCanonicalArray(PCanonicalArray(PInt32(true), true), true))
    altrBranch = In(1, PCanonicalArray(PCanonicalArray(PInt32(false), true), true))

    ir = If(True(), cnsqBranch, altrBranch)
    assertPType(ir, PCanonicalArray(PCanonicalArray(PInt32(false), true), true))

    cnsqBranch = In(0, PCanonicalArray(PCanonicalArray(PInt32(true), false), true))
    altrBranch = In(1, PCanonicalArray(PCanonicalArray(PInt32(false), true), true))

    ir = If(True(), cnsqBranch, altrBranch)
    assertPType(ir, PCanonicalArray(PCanonicalArray(PInt32(false), false), true))
  }

  @Test def testLet() {
    assertEvalsTo(Let("v", I32(5), Ref("v", TInt32)), 5)
    assertEvalsTo(Let("v", NA(TInt32), Ref("v", TInt32)), null)
    assertEvalsTo(Let("v", I32(5), NA(TInt32)), null)
    assertEvalsTo(ToArray(StreamMap(Let("v", I32(5), StreamRange(0, Ref("v", TInt32), 1)), "x", Ref("x", TInt32) + I32(2))),
      FastIndexedSeq(2, 3, 4, 5, 6))
    assertEvalsTo(
      ToArray(StreamMap(Let("q", I32(2),
      StreamMap(Let("v", Ref("q", TInt32) + I32(3),
        StreamRange(0, Ref("v", TInt32), 1)),
        "x", Ref("x", TInt32) + Ref("q", TInt32))),
        "y", Ref("y", TInt32) + I32(3))),
      FastIndexedSeq(5, 6, 7, 8, 9))

    // test let binding streams
    assertEvalsTo(Let("s", MakeStream(Seq(I32(0), I32(5)), TStream(TInt32)), ToArray(Ref("s", TStream(TInt32)))),
                  FastIndexedSeq(0, 5))
    assertEvalsTo(Let("s", NA(TStream(TInt32)), ToArray(Ref("s", TStream(TInt32)))),
                  null)
    assertEvalsTo(
      ToArray(Let("s",
                  MakeStream(Seq(I32(0), I32(5)), TStream(TInt32)),
                  StreamTake(Ref("s", TStream(TInt32)), I32(1)))),
      FastIndexedSeq(0))
  }

  @Test def testMakeArray() {
    assertEvalsTo(MakeArray(FastSeq(I32(5), NA(TInt32), I32(-3)), TArray(TInt32)), FastIndexedSeq(5, null, -3))
    assertEvalsTo(MakeArray(FastSeq(), TArray(TInt32)), FastIndexedSeq())
  }

  @Test def testMakeArrayInferPTypeFromNestedRef() {
    var ir = MakeArray(FastSeq(), TArray(TInt32))
    assertPType(ir, PCanonicalArray(PInt32(true), true))

    val eltType = TStruct("a" -> TArray(TArray(TInt32)), "b" -> TInt32, "c" -> TDict(TInt32, TString))

    val pTypes = Array[PType](
      PCanonicalStruct(true,
        "a" -> PCanonicalArray(PCanonicalArray(PInt32(false), true), false),
        "b" -> PInt32(true),
        "c" -> PCanonicalDict(PInt32(false), PCanonicalString(false), false)),
      PCanonicalStruct(true,
        "a" -> PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
        "b" -> PInt32(true),
        "c" -> PCanonicalDict(PInt32(true), PCanonicalString(true), true)))

    val unified = PCanonicalStruct(true,
      "a" -> PCanonicalArray(PCanonicalArray(PInt32(false), true), false),
      "b" -> PInt32(true),
      "c" -> PCanonicalDict(PInt32(false), PCanonicalString(false), false))

    assertPType(MakeArray(Array(In(0, pTypes(0))), TArray(eltType)), PCanonicalArray(pTypes(0), true))
    assertPType(MakeArray(Array(In(0, pTypes(0)), In(1, pTypes(1))), TArray(eltType)), PCanonicalArray(pTypes(0), true))
  }

  @Test def testMakeArrayInferPType() {
    var ir = MakeArray(FastSeq(I32(5), NA(TInt32), I32(-3)), TArray(TInt32))

    assertPType(ir, PCanonicalArray(PInt32(false), true))

    ir = MakeArray(FastSeq(I32(5), I32(1), I32(-3)), TArray(TInt32))

    assertPType(ir, PCanonicalArray(PInt32(true), true))

    ir = MakeArray(FastSeq(I32(5), I32(1), I32(-3)), TArray(TInt32))
  }

  @Test def testGetNestedElementPTypesI32() {
    var types = Seq(PInt32(true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PInt32(true))

    types = Seq(PInt32(false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt32(false))

    types = Seq(PInt32(false), PInt32(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt32(false))

    types = Seq(PInt32(true), PInt32(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt32(true))
  }

  @Test def testGetNestedElementPTypesI64() {
    var types = Seq(PInt64(true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PInt64(true))

    types = Seq(PInt64(false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt64(false))

    types = Seq(PInt64(false), PInt64(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt64(false))

    types = Seq(PInt64(true), PInt64(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PInt64(true))
  }

  @Test def testGetNestedElementPFloat32() {
    var types = Seq(PFloat32(true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(true))

    types = Seq(PFloat32(false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(false))

    types = Seq(PFloat32(false), PFloat32(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(false))

    types = Seq(PFloat32(true), PFloat32(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat32(true))
  }

  @Test def testGetNestedElementPFloat64() {
    var types = Seq(PFloat64(true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(true))

    types = Seq(PFloat64(false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(false))

    types = Seq(PFloat64(false), PFloat64(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(false))

    types = Seq(PFloat64(true), PFloat64(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PFloat64(true))
  }

  @Test def testGetNestedElementPCanonicalString() {
    var types = Seq(PCanonicalString(true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(true))

    types = Seq(PCanonicalString(false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(false))

    types = Seq(PCanonicalString(false), PCanonicalString(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(false))

    types = Seq(PCanonicalString(true), PCanonicalString(true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalString(true))
  }

  @Test def testGetNestedPCanonicalArray() {
    var types = Seq(PCanonicalArray(PInt32(true), true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), true))

    types = Seq(PCanonicalArray(PInt32(true), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), false))

    types = Seq(PCanonicalArray(PInt32(false), true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), true))

    types = Seq(PCanonicalArray(PInt32(false), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), false))

    types = Seq(
      PCanonicalArray(PInt32(true), true),
      PCanonicalArray(PInt32(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(true), true))

    types = Seq(
      PCanonicalArray(PInt32(false), true),
      PCanonicalArray(PInt32(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), true))

    types = Seq(
      PCanonicalArray(PInt32(false), true),
      PCanonicalArray(PInt32(true), false)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PInt32(false), false))

    types = Seq(
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(true), true), true))

    types = Seq(
      PCanonicalArray(PCanonicalArray(PInt32(true), true), true),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), true), true))

    types = Seq(
      PCanonicalArray(PCanonicalArray(PInt32(true), false), true),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), false), true))

    types = Seq(
      PCanonicalArray(PCanonicalArray(PInt32(true), false), false),
      PCanonicalArray(PCanonicalArray(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalArray(PCanonicalArray(PInt32(false), false), false))
  }

  @Test def testGetNestedPStream() {
    var types = Seq(PCanonicalStream(PInt32(true), true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(true), true))

    types = Seq(PCanonicalStream(PInt32(true), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(true), false))

    types = Seq(PCanonicalStream(PInt32(false), true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(false), true))

    types = Seq(PCanonicalStream(PInt32(false), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(false), false))

    types = Seq(
      PCanonicalStream(PInt32(true), true),
      PCanonicalStream(PInt32(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(true), true))

    types = Seq(
      PCanonicalStream(PInt32(false), true),
      PCanonicalStream(PInt32(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(false), true))

    types = Seq(
      PCanonicalStream(PInt32(false), true),
      PCanonicalStream(PInt32(true), false)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PInt32(false), false))

    types = Seq(
      PCanonicalStream(PCanonicalStream(PInt32(true), true), true),
      PCanonicalStream(PCanonicalStream(PInt32(true), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PCanonicalStream(PInt32(true), true), true))

    types = Seq(
      PCanonicalStream(PCanonicalStream(PInt32(true), true), true),
      PCanonicalStream(PCanonicalStream(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PCanonicalStream(PInt32(false), true), true))

    types = Seq(
      PCanonicalStream(PCanonicalStream(PInt32(true), false), true),
      PCanonicalStream(PCanonicalStream(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PCanonicalStream(PInt32(false), false), true))

    types = Seq(
      PCanonicalStream(PCanonicalStream(PInt32(true), false), false),
      PCanonicalStream(PCanonicalStream(PInt32(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStream(PCanonicalStream(PInt32(false), false), false))
  }

  @Test def testGetNestedElementPCanonicalDict() {
    var types = Seq(PCanonicalDict(PInt32(true), PCanonicalString(true), true))
    var res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), true))

    types = Seq(PCanonicalDict(PInt32(false), PCanonicalString(true), true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(true), true))

    types = Seq(PCanonicalDict(PInt32(true), PCanonicalString(false), true))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(false), true))

    types = Seq(PCanonicalDict(PInt32(true), PCanonicalString(true), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), false))

    types = Seq(PCanonicalDict(PInt32(false), PCanonicalString(false), false))
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), false))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), true))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalString(true), false),
      PCanonicalDict(PInt32(true), PCanonicalString(true), false)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalString(true), false))

    types = Seq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(true), true))

    types = Seq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), true),
      PCanonicalDict(PInt32(true), PCanonicalString(false), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), true))

    types = Seq(
      PCanonicalDict(PInt32(false), PCanonicalString(true), false),
      PCanonicalDict(PInt32(true), PCanonicalString(false), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(false), PCanonicalString(false), false))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(true), true), true),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(true), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(true), true), true))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(true), true), true),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(false), true), true))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(true), true), true),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(false), true), true))

    types = Seq(
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(true), false), true),
      PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(true), PCanonicalString(false), true), true)
    )
    res  = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalDict(PInt32(true), PCanonicalDict(PInt32(false), PCanonicalString(false), false), true))
  }

  @Test def testGetNestedElementPCanonicalStruct() {
    var types = Seq(PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = Seq(PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = Seq(PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(true)))

    types = Seq(PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(false)))

    types = Seq(PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = Seq(
      PCanonicalStruct(false, "a" -> PInt32(true), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(false))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(false, "a" -> PInt32(false), "b" -> PInt32(false)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)),"b" -> PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)), "b" -> PInt32(true)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(true)),"b" -> PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(true)), "b" -> PInt32(true)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(false)), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(false), "d" -> PInt32(false)), "b" -> PInt32(true)))

    types = Seq(
      PCanonicalStruct(true, "a" -> PCanonicalStruct(false, "c" -> PInt32(false), "d" -> PInt32(false)), "b" -> PInt32(true)),
      PCanonicalStruct(true, "a" -> PCanonicalStruct(true, "c" -> PInt32(true), "d" -> PInt32(true)), "b" -> PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalStruct(true, "a" -> PCanonicalStruct(false, "c" -> PInt32(false), "d" -> PInt32(false)), "b" -> PInt32(true)))
  }

  @Test def testGetNestedElementPCanonicalTuple() {
    var types = Seq(PCanonicalTuple(true, PInt32(true)))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(true)))

    types = Seq(PCanonicalTuple(false, PInt32(true)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(true)))

    types = Seq(PCanonicalTuple(true, PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(false)))

    types = Seq(PCanonicalTuple(false, PInt32(false)))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(false)))

    types = Seq(
      PCanonicalTuple(true, PInt32(true)),
      PCanonicalTuple(true, PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PInt32(true)))

    types = Seq(
      PCanonicalTuple(true, PInt32(true)),
      PCanonicalTuple(false, PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(true)))

    types = Seq(
      PCanonicalTuple(true, PInt32(false)),
      PCanonicalTuple(false, PInt32(true))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(false, PInt32(false)))

    types = Seq(
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(true))),
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false)))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false))))

    types = Seq(
      PCanonicalTuple(true, PCanonicalTuple(false, PInt32(true))),
      PCanonicalTuple(true, PCanonicalTuple(true, PInt32(false)))
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalTuple(true, PCanonicalTuple(false, PInt32(false))))
  }

  @Test def testGetNestedElementPCanonicalSet() {
    var types = Seq(PCanonicalSet(PInt32(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), true))

    types = Seq(PCanonicalSet(PInt32(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), false))

    types = Seq(PCanonicalSet(PInt32(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), true))

    types = Seq(PCanonicalSet(PInt32(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), false))

    types = Seq(
      PCanonicalSet(PInt32(true), true),
      PCanonicalSet(PInt32(true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(true), true))

    types = Seq(
      PCanonicalSet(PInt32(false), true),
      PCanonicalSet(PInt32(true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), true))

    types = Seq(
      PCanonicalSet(PInt32(false), true),
      PCanonicalSet(PInt32(true), false)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PInt32(false), false))

    types = Seq(
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true),
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(true), true), true))

    types = Seq(
      PCanonicalSet(PCanonicalSet(PInt32(true), true), true),
      PCanonicalSet(PCanonicalSet(PInt32(false), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(false), true), true))

    types = Seq(
      PCanonicalSet(PCanonicalSet(PInt32(true), false), true),
      PCanonicalSet(PCanonicalSet(PInt32(false), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalSet(PCanonicalSet(PInt32(false), false), true))
  }

  @Test def testGetNestedElementPCanonicalInterval() {
    var types = Seq(PCanonicalInterval(PInt32(true), true))
    var res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), true))

    types = Seq(PCanonicalInterval(PInt32(true), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), false))

    types = Seq(PCanonicalInterval(PInt32(false), true))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), true))

    types = Seq(PCanonicalInterval(PInt32(false), false))
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), false))

    types = Seq(
      PCanonicalInterval(PInt32(true), true),
      PCanonicalInterval(PInt32(true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), true))

    types = Seq(
      PCanonicalInterval(PInt32(false), true),
      PCanonicalInterval(PInt32(true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), true))

    types = Seq(
      PCanonicalInterval(PInt32(true), true),
      PCanonicalInterval(PInt32(true), false)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(true), false))

    types = Seq(
      PCanonicalInterval(PInt32(false), true),
      PCanonicalInterval(PInt32(true), false)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PInt32(false), false))

    types = Seq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true))

    types = Seq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true))

    types = Seq(
      PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(true), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true))

    types = Seq(
      PCanonicalInterval(PCanonicalInterval(PInt32(true), false), true),
      PCanonicalInterval(PCanonicalInterval(PInt32(false), true), true)
    )
    res = InferPType.getCompatiblePType(types)
    assert(res == PCanonicalInterval(PCanonicalInterval(PInt32(false), false), true))
  }

  @Test def testToDictInferPtype() {
    val allRequired = ToDict(MakeStream(FastIndexedSeq(
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("a"))),
      MakeTuple.ordered(FastIndexedSeq(I32(10), Str("b")))
    ), TStream(TTuple(TInt32, TString))))

    assertPType(allRequired, PCanonicalDict(PInt32(true), PCanonicalString(true), true))

    var notAllRequired = ToDict(MakeStream(FastIndexedSeq(
      MakeTuple.ordered(FastIndexedSeq(NA(TInt32), Str("a"))),
      MakeTuple.ordered(FastIndexedSeq(I32(10), Str("b")))
    ), TStream(TTuple(TInt32, TString))))

    assertPType(notAllRequired, PCanonicalDict(PInt32(false), PCanonicalString(true), true))

    notAllRequired = ToDict(MakeStream(FastIndexedSeq(
      MakeTuple.ordered(FastIndexedSeq(NA(TInt32), Str("a"))),
      MakeTuple.ordered(FastIndexedSeq(I32(10), NA(TString))
    )), TStream(TTuple(TInt32, TString))))

    assertPType(notAllRequired, PCanonicalDict(PInt32(false), PCanonicalString(false), true))
  }

  @Test def testMakeStruct() {
    assertEvalsTo(MakeStruct(FastSeq()), Row())
    assertEvalsTo(MakeStruct(FastSeq("a" -> NA(TInt32), "b" -> 4, "c" -> 0.5)), Row(null, 4, 0.5))
    //making sure wide structs get emitted without failure
    assertEvalsTo(GetField(MakeStruct((0 until 20000).map(i => s"foo$i" -> I32(1))), "foo1"), 1)
  }

  @Test def testMakeStructInferPType() {
    var ir = MakeStruct(FastSeq())
    assertPType(ir, PCanonicalStruct(true))

    ir = MakeStruct(FastSeq("a" -> NA(TInt32), "b" -> 4, "c" -> 0.5))
    assertPType(ir, PCanonicalStruct(true, "a" -> PInt32(false), "b" -> PInt32(true), "c" -> PFloat64(true)))

    val ir2 = GetField(MakeStruct((0 until 20000).map(i => s"foo$i" -> I32(1))), "foo1")
    assertPType(ir2, PInt32(true))
  }

  @Test def testMakeArrayWithDifferentRequiredness(): Unit = {
    val pt1 = PCanonicalArray(PCanonicalStruct("a" -> PInt32(), "b" -> PCanonicalArray(PInt32())))
    val pt2 = PCanonicalArray(PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PCanonicalArray(PInt32(), true)))

    val value = Row(2, FastIndexedSeq(1))
    assertEvalsTo(
      MakeArray(Seq(In(0, pt1.elementType), In(1, pt2.elementType)), pt1.virtualType),
      FastIndexedSeq((null, pt1.virtualType.elementType), (value, pt2.virtualType.elementType)),
      FastIndexedSeq(null, value)
    )
  }

  @Test def testMakeTuple() {
    assertEvalsTo(MakeTuple.ordered(FastSeq()), Row())
    assertEvalsTo(MakeTuple.ordered(FastSeq(NA(TInt32), 4, 0.5)), Row(null, 4, 0.5))
    //making sure wide structs get emitted without failure
    assertEvalsTo(GetTupleElement(MakeTuple.ordered((0 until 20000).map(I32)), 1), 1)
  }

  @Test def testGetTupleElement() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = MakeTuple.ordered(FastIndexedSeq(I32(5), Str("abc"), NA(TInt32)))
    val na = NA(TTuple(TInt32, TString))

    assertEvalsTo(GetTupleElement(t, 0), 5)
    assertEvalsTo(GetTupleElement(t, 1), "abc")
    assertEvalsTo(GetTupleElement(t, 2), null)
    assertEvalsTo(GetTupleElement(na, 0), null)
  }

    @Test def testLetBoundPrunedTuple(): Unit = {
    implicit val execStrats = ExecStrategy.unoptimizedCompileOnly
    val t2 = MakeTuple(FastSeq((2, I32(5))))

    val letBoundTuple = bindIR(t2) { tupleRef =>
      GetTupleElement(tupleRef, 2)
    }

    assertEvalsTo(letBoundTuple, 5)
  }

  @Test def testArrayRef() {
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32)), TArray(TInt32)), I32(0)), 5)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32)), TArray(TInt32)), I32(1)), null)
    assertEvalsTo(ArrayRef(MakeArray(FastIndexedSeq(I32(5), NA(TInt32)), TArray(TInt32)), NA(TInt32)), null)

    assertFatal(ArrayRef(MakeArray(FastIndexedSeq(I32(5)), TArray(TInt32)), I32(2)), "array index out of bounds")
  }

  @Test def testArrayLen() {
    assertEvalsTo(ArrayLen(NA(TArray(TInt32))), null)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(), TArray(TInt32))), 0)
    assertEvalsTo(ArrayLen(MakeArray(FastIndexedSeq(I32(5), NA(TInt32)), TArray(TInt32))), 2)
  }

  @Test def testArraySort() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ArraySort(ToStream(NA(TArray(TInt32)))), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32), I32(2)), TArray(TInt32))
    assertEvalsTo(ArraySort(ToStream(a)),
      FastIndexedSeq(-7, 2, 2, null))
    assertEvalsTo(ArraySort(ToStream(a), False()),
      FastIndexedSeq(2, 2, -7, null))
  }

  @Test def testStreamZip() {
    val range12 = StreamRange(0, 12, 1)
    val range6 = StreamRange(0, 12, 2)
    val range8 = StreamRange(0, 24, 3)
    val empty = StreamRange(0, 0, 1)
    val lit6 = ToStream(Literal(TArray(TFloat64), FastIndexedSeq(0d, -1d, 2.5d, -3d, 4d, null)))
    val range6dup = StreamRange(0, 6, 1)

    def zip(behavior: ArrayZipBehavior, irs: IR*): IR = StreamZip(
      irs.toFastIndexedSeq,
      irs.indices.map(_.toString),
      MakeTuple.ordered(irs.zipWithIndex.map { case (ir, i) => Ref(i.toString, ir.typ.asInstanceOf[TStream].elementType) }),
      behavior
    )
    def zipToTuple(behavior: ArrayZipBehavior, irs: IR*): IR = ToArray(zip(behavior, irs: _*))

    for (b <- Array(ArrayZipBehavior.TakeMinLength, ArrayZipBehavior.ExtendNA)) {
      assertEvalSame(zipToTuple(b, range12), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range6, range8), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range6, range8), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range6, range8, lit6), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range12, lit6), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range12, lit6, empty), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, empty, lit6), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, empty), FastIndexedSeq())
    }

    for (b <- Array(ArrayZipBehavior.AssumeSameLength, ArrayZipBehavior.AssertSameLength)) {
      assertEvalSame(zipToTuple(b, range6, lit6), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range6, lit6, range6dup), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, range12), FastIndexedSeq())
      assertEvalSame(zipToTuple(b, empty), FastIndexedSeq())
    }

    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.TakeMinLength, range6, range8)), 6)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.ExtendNA, range6, range8)), 8)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.AssertSameLength, range8, range8)), 8)
    assertEvalsTo(StreamLen(zip(ArrayZipBehavior.AssumeSameLength, range8, range8)), 8)

    // https://github.com/hail-is/hail/issues/8359
    is.hail.TestUtils.assertThrows[HailException](zipToTuple(ArrayZipBehavior.AssertSameLength, range6, range8): IR, "zip: length mismatch": String)
    is.hail.TestUtils.assertThrows[HailException](zipToTuple(ArrayZipBehavior.AssertSameLength, range12, lit6): IR, "zip: length mismatch": String)
  }

  @Test def testToSet() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToSet(ToStream(NA(TArray(TInt32)))), null)
    assertEvalsTo(ToSet(NA(TStream(TInt32))), null)

    val a = MakeArray(FastIndexedSeq(I32(-7), I32(2), NA(TInt32), I32(2)), TArray(TInt32))
    assertEvalsTo(ToSet(ToStream(a)), Set(-7, 2, null))
  }

  @Test def testToArrayFromSet() {
    val t = TSet(TInt32)
    assertEvalsTo(CastToArray(NA(t)), null)
    assertEvalsTo(CastToArray(In(0, t)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      FastIndexedSeq(-7, 2, null))
  }

  @Test def testToDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    assertEvalsTo(ToDict(ToStream(NA(TArray(TTuple(FastIndexedSeq(TInt32, TString): _*))))), null)

    val a = MakeArray(FastIndexedSeq(
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("a"))),
      MakeTuple.ordered(FastIndexedSeq(I32(5), Str("a"))), // duplicate key-value pair
      MakeTuple.ordered(FastIndexedSeq(NA(TInt32), Str("b"))),
      MakeTuple.ordered(FastIndexedSeq(I32(3), NA(TString))),
      NA(TTuple(FastIndexedSeq(TInt32, TString): _*)) // missing value
    ), TArray(TTuple(FastIndexedSeq(TInt32, TString): _*)))

    assertEvalsTo(ToDict(ToStream(a)), Map(5 -> "a", (null, "b"), 3 -> null))
  }

  @Test def testToArrayFromDict() {
    val t = TDict(TInt32, TString)
    assertEvalsTo(CastToArray(NA(t)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(CastToArray(In(0, t)),
      // wtf you can't do null -> ...
      FastIndexedSeq((d, t)),
      FastIndexedSeq(Row(1, "a"), Row(2, null), Row(null, "c")))
  }

  @Test def testToArrayFromArray() {
    val t = TArray(TInt32)
    assertEvalsTo(NA(t), null)
    assertEvalsTo(In(0, t),
      FastIndexedSeq((FastIndexedSeq(-7, 2, null, 2), t)),
      FastIndexedSeq(-7, 2, null, 2))
  }

  @Test def testSetContains() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TSet(TInt32)
    assertEvalsTo(invoke("contains", TBoolean, NA(t), I32(2)), null)

    assertEvalsTo(invoke("contains", TBoolean, In(0, t), NA(TInt32)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(2)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(0)),
      FastIndexedSeq((Set(-7, 2, null), t)),
      false)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(7)),
      FastIndexedSeq((Set(-7, 2), t)),
      false)
  }

  @Test def testDictContains() {
    implicit val execStrats = ExecStrategy.javaOnly

    val t = TDict(TInt32, TString)
    assertEvalsTo(invoke("contains", TBoolean, NA(t), I32(2)), null)

    val d = Map(1 -> "a", 2 -> null, (null, "c"))
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), NA(TInt32)),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(2)),
      FastIndexedSeq((d, t)),
      true)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(0)),
      FastIndexedSeq((d, t)),
      false)
    assertEvalsTo(invoke("contains", TBoolean, In(0, t), I32(3)),
      FastIndexedSeq((Map(1 -> "a", 2 -> null), t)),
      false)
  }

  @Test def testLowerBoundOnOrderedCollectionArray() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TArray(TInt32))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val awoutna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4)), TArray(TInt32))
    val awna = MakeArray(FastIndexedSeq(I32(0), I32(2), I32(4), NA(TInt32)), TArray(TInt32))
    val awdups = MakeArray(FastIndexedSeq(I32(0), I32(0), I32(2), I32(4), I32(4), NA(TInt32)), TArray(TInt32))
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
        (LowerBoundOnOrderedCollection(awdups, I32(4), onKey = false), 3)
    )
  }

  @Test def testLowerBoundOnOrderedCollectionSet() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TSet(TInt32))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = false), null)

    val swoutna = ToSet(MakeStream(FastIndexedSeq(I32(0), I32(2), I32(4), I32(4)), TStream(TInt32)))
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(-1), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(0), onKey = false), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(1), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(2), onKey = false), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(3), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(4), onKey = false), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, I32(5), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swoutna, NA(TInt32), onKey = false), 3)

    val swna = ToSet(MakeStream(FastIndexedSeq(I32(0), I32(2), I32(2), I32(4), NA(TInt32)), TStream(TInt32)))
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, NA(TInt32), onKey = false), 3)
    assertEvalsTo(LowerBoundOnOrderedCollection(swna, I32(5), onKey = false), 3)
  }

  @Test def testLowerBoundOnOrderedCollectionDict() {
    implicit val execStrats = ExecStrategy.javaOnly

    val na = NA(TDict(TInt32, TString))
    assertEvalsTo(LowerBoundOnOrderedCollection(na, I32(0), onKey = true), null)

    val dwna = TestUtils.IRDict((1, 3), (3, null), (null, 5))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(2), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(3), onKey = true), 1)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, I32(5), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwna, NA(TInt32), onKey = true), 2)

    val dwoutna = TestUtils.IRDict((1, 3), (3, null))
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(-1), onKey = true), 0)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, I32(4), onKey = true), 2)
    assertEvalsTo(LowerBoundOnOrderedCollection(dwoutna, NA(TInt32), onKey = true), 2)
  }

  @Test def testStreamLen(): Unit = {
    val a = StreamLen(MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32)))
    assertEvalsTo(a, 3)

    val missing = StreamLen(NA(TStream(TInt64)))
    assertEvalsTo(missing, null)

    val range1 = StreamLen(StreamRange(1, 11, 1))
    assertEvalsTo(range1, 10)
    val range2 = StreamLen(StreamRange(20, 40, 2))
    assertEvalsTo(range2, 10)
    val range3 = StreamLen(StreamRange(-10, 5, 1))
    assertEvalsTo(range3, 15)
    val mappedRange = StreamLen(mapIR(StreamRange(2, 7, 1)) { ref => maxIR(ref, 3)})
    assertEvalsTo(mappedRange, 5)

    val streamOfStreams = mapIR(rangeIR(5)) { elementRef => rangeIR(elementRef) }
    assertEvalsTo(StreamLen(flatMapIR(streamOfStreams){ x => x}), 4 + 3 + 2 + 1)

    val filteredRange = StreamLen(StreamFilter(rangeIR(12), "x", irToPrimitiveIR(Ref("x", TInt32)) < 5))
    assertEvalsTo(filteredRange, 5)
  }

  @Test def testStreamTake() {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamTake(naa, I32(2))), null)
    assertEvalsTo(ToArray(StreamTake(a, NA(TInt32))), null)
    assertEvalsTo(ToArray(StreamTake(a, I32(0))), FastIndexedSeq())
    assertEvalsTo(ToArray(StreamTake(a, I32(2))), FastIndexedSeq(3, null))
    assertEvalsTo(ToArray(StreamTake(a, I32(5))), FastIndexedSeq(3, null, 7))
    assertFatal(ToArray(StreamTake(a, I32(-1))), "StreamTake: negative length")
    assertEvalsTo(StreamLen(StreamTake(a, 2)), 2)
  }

  @Test def testStreamDrop() {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamDrop(naa, I32(2))), null)
    assertEvalsTo(ToArray(StreamDrop(a, NA(TInt32))), null)
    assertEvalsTo(ToArray(StreamDrop(a, I32(0))), FastIndexedSeq(3, null, 7))
    assertEvalsTo(ToArray(StreamDrop(a, I32(2))), FastIndexedSeq(7))
    assertEvalsTo(ToArray(StreamDrop(a, I32(5))), FastIndexedSeq())
    assertFatal(ToArray(StreamDrop(a, I32(-1))), "StreamDrop: negative num")

    assertEvalsTo(StreamLen(StreamDrop(a, 1)), 2)
  }

  def toNestedArray(stream: IR): IR = {
    val innerType = coerce[TStream](coerce[TStream](stream.typ).elementType)
    ToArray(StreamMap(stream, "inner", ToArray(Ref("inner", innerType))))
  }

  @Test def testStreamGrouped() {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(toNestedArray(StreamGrouped(naa, I32(2))), null)
    assertEvalsTo(toNestedArray(StreamGrouped(a, NA(TInt32))), null)
    assertEvalsTo(toNestedArray(StreamGrouped(MakeStream(Seq(), TStream(TInt32)), I32(2))), FastIndexedSeq())
    assertEvalsTo(toNestedArray(StreamGrouped(a, I32(1))), FastIndexedSeq(FastIndexedSeq(3), FastIndexedSeq(null), FastIndexedSeq(7)))
    assertEvalsTo(toNestedArray(StreamGrouped(a, I32(2))), FastIndexedSeq(FastIndexedSeq(3, null), FastIndexedSeq(7)))
    assertEvalsTo(toNestedArray(StreamGrouped(a, I32(5))), FastIndexedSeq(FastIndexedSeq(3, null, 7)))
    assertFatal(toNestedArray(StreamGrouped(a, I32(0))), "StreamGrouped: nonpositive size")

    val r = rangeIR(10)

    // test when inner streams are unused
    assertEvalsTo(streamForceCount(StreamGrouped(rangeIR(10), 2)), 5)

    assertEvalsTo(StreamLen(StreamGrouped(r, 2)), 5)

    def takeFromEach(stream: IR, take: IR, fromEach: IR): IR = {
      val innerType = coerce[TStream](stream.typ)
      StreamMap(StreamGrouped(stream, fromEach), "inner", StreamTake(Ref("inner", innerType), take))
    }

    assertEvalsTo(toNestedArray(takeFromEach(r, I32(1), I32(3))),
                  FastIndexedSeq(FastIndexedSeq(0), FastIndexedSeq(3), FastIndexedSeq(6), FastIndexedSeq(9)))
    assertEvalsTo(toNestedArray(takeFromEach(r, I32(2), I32(3))),
                  FastIndexedSeq(FastIndexedSeq(0, 1), FastIndexedSeq(3, 4), FastIndexedSeq(6, 7), FastIndexedSeq(9)))
    assertEvalsTo(toNestedArray(takeFromEach(r, I32(0), I32(5))),
                  FastIndexedSeq(FastIndexedSeq(), FastIndexedSeq()))
  }

  @Test def testStreamGroupByKey() {
    val structType = TStruct("a" -> TInt32, "b" -> TInt32)
    val naa = NA(TStream(structType))
    val a = MakeStream(
      Seq(
        MakeStruct(Seq("a" -> I32(3), "b" -> I32(1))),
        MakeStruct(Seq("a" -> I32(3), "b" -> I32(3))),
        MakeStruct(Seq("a" -> NA(TInt32), "b" -> I32(-1))),
        MakeStruct(Seq("a" -> NA(TInt32), "b" -> I32(-2))),
        MakeStruct(Seq("a" -> I32(1), "b" -> I32(2))),
        MakeStruct(Seq("a" -> I32(1), "b" -> I32(4))),
        MakeStruct(Seq("a" -> I32(1), "b" -> I32(6))),
        MakeStruct(Seq("a" -> I32(4), "b" -> NA(TInt32)))),
      TStream(structType))

    def group(a: IR): IR = StreamGroupByKey(a, FastIndexedSeq("a"))
    assertEvalsTo(toNestedArray(group(naa)), null)
    assertEvalsTo(toNestedArray(group(a)),
                  FastIndexedSeq(FastIndexedSeq(Row(3, 1), Row(3, 3)),
                                 FastIndexedSeq(Row(null, -1)),
                                 FastIndexedSeq(Row(null, -2)),
                                 FastIndexedSeq(Row(1, 2), Row(1, 4), Row(1, 6)),
                                 FastIndexedSeq(Row(4, null))))
    assertEvalsTo(toNestedArray(group(MakeStream(Seq(), TStream(structType)))), FastIndexedSeq())

    // test when inner streams are unused
    assertEvalsTo(streamForceCount(group(a)), 5)

    def takeFromEach(stream: IR, take: IR): IR = {
      val innerType = coerce[TStream](stream.typ)
      StreamMap(group(stream), "inner", StreamTake(Ref("inner", innerType), take))
    }

    assertEvalsTo(toNestedArray(takeFromEach(a, I32(1))),
                  FastIndexedSeq(FastIndexedSeq(Row(3, 1)),
                                 FastIndexedSeq(Row(null, -1)),
                                 FastIndexedSeq(Row(null, -2)),
                                 FastIndexedSeq(Row(1, 2)),
                                 FastIndexedSeq(Row(4, null))))
    assertEvalsTo(toNestedArray(takeFromEach(a, I32(2))),
                  FastIndexedSeq(FastIndexedSeq(Row(3, 1), Row(3, 3)),
                                 FastIndexedSeq(Row(null, -1)),
                                 FastIndexedSeq(Row(null, -2)),
                                 FastIndexedSeq(Row(1, 2), Row(1, 4)),
                                 FastIndexedSeq(Row(4, null))))
  }

  @Test def testStreamMap() {
    val naa = NA(TStream(TInt32))
    val a = MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamMap(naa, "a", I32(5))), null)

    assertEvalsTo(ToArray(StreamMap(a, "a", ApplyBinaryPrimOp(Add(), Ref("a", TInt32), I32(1)))), FastIndexedSeq(4, null, 8))

    assertEvalsTo(ToArray(Let("a", I32(5),
      StreamMap(a, "a", Ref("a", TInt32)))),
      FastIndexedSeq(3, null, 7))
  }

  @Test def testStreamFilter() {
    val nsa = NA(TStream(TInt32))
    val a = MakeStream(Seq(I32(3), NA(TInt32), I32(7)), TStream(TInt32))

    assertEvalsTo(ToArray(StreamFilter(nsa, "x", True())), null)

    assertEvalsTo(ToArray(StreamFilter(a, "x", NA(TBoolean))), FastIndexedSeq())
    assertEvalsTo(ToArray(StreamFilter(a, "x", False())), FastIndexedSeq())
    assertEvalsTo(ToArray(StreamFilter(a, "x", True())), FastIndexedSeq(3, null, 7))

    assertEvalsTo(ToArray(StreamFilter(a, "x",
      IsNA(Ref("x", TInt32)))), FastIndexedSeq(null))
    assertEvalsTo(ToArray(StreamFilter(a, "x",
      ApplyUnaryPrimOp(Bang(), IsNA(Ref("x", TInt32))))), FastIndexedSeq(3, 7))

    assertEvalsTo(ToArray(StreamFilter(a, "x",
      ApplyComparisonOp(LT(TInt32), Ref("x", TInt32), I32(6)))), FastIndexedSeq(3))
  }

  @Test def testArrayFlatMap() {
    val ta = TArray(TInt32)
    val ts = TStream(TInt32)
    val tsa = TStream(ta)
    val nsa = NA(tsa)
    val naas = MakeStream(FastIndexedSeq(NA(ta), NA(ta)), tsa)
    val a = MakeStream(FastIndexedSeq(
      MakeArray(FastIndexedSeq(I32(7), NA(TInt32)), ta),
      NA(ta),
      MakeArray(FastIndexedSeq(I32(2)), ta)),
      tsa)

    assertEvalsTo(ToArray(StreamFlatMap(nsa, "a", MakeStream(FastIndexedSeq(I32(5)), ts))), null)

    assertEvalsTo(ToArray(StreamFlatMap(naas, "a", ToStream(Ref("a", ta)))), FastIndexedSeq())

    assertEvalsTo(ToArray(StreamFlatMap(a, "a", ToStream(Ref("a", ta)))), FastIndexedSeq(7, null, 2))

    assertEvalsTo(ToArray(StreamFlatMap(StreamRange(I32(0), I32(3), I32(1)), "i", ToStream(ArrayRef(ToArray(a), Ref("i", TInt32))))), FastIndexedSeq(7, null, 2))

    assertEvalsTo(ToArray(Let("a", I32(5), StreamFlatMap(a, "a", ToStream(Ref("a", ta))))), FastIndexedSeq(7, null, 2))

    val b = MakeStream(FastIndexedSeq(
      MakeArray(FastIndexedSeq(I32(7), I32(0)), ta),
      NA(ta),
      MakeArray(FastIndexedSeq(I32(2)), ta)),
      tsa)
    assertEvalsTo(ToArray(Let("a", I32(5), StreamFlatMap(b, "b", ToStream(Ref("b", ta))))), FastIndexedSeq(7, 0, 2))

    val st = MakeStream(List(I32(1), I32(5), I32(2), NA(TInt32)), TStream(TInt32))
    val expected = FastIndexedSeq(-1, 0, -1, 0, 1, 2, 3, 4, -1, 0, 1)
    assertEvalsTo(ToArray(StreamFlatMap(st, "foo", StreamRange(I32(-1), Ref("foo", TInt32), I32(1)))), expected)
  }

  @Test def testStreamFold() {
    def fold(s: IR, zero: IR, f: (IR, IR) => IR): IR =
      StreamFold(s, zero, "_accum", "_elt", f(Ref("_accum", zero.typ), Ref("_elt", zero.typ)))

    assertEvalsTo(fold(StreamRange(1, 2, 1), NA(TBoolean), (accum, elt) => IsNA(accum)), true)
    assertEvalsTo(fold(TestUtils.IRStream(1, 2, 3), 0, (accum, elt) => accum + elt), 6)
    assertEvalsTo(fold(TestUtils.IRStream(1, 2, 3), NA(TInt32), (accum, elt) => accum + elt), null)
    assertEvalsTo(fold(TestUtils.IRStream(1, null, 3), NA(TInt32), (accum, elt) => accum + elt), null)
    assertEvalsTo(fold(TestUtils.IRStream(1, null, 3), 0, (accum, elt) => accum + elt), null)
    assertEvalsTo(fold(TestUtils.IRStream(1, null, 3), NA(TInt32), (accum, elt) => I32(5) + I32(5)), 10)
  }

  @Test def testArrayFold2() {
    implicit val execStrats = ExecStrategy.compileOnly

    val af = StreamFold2(ToStream(In(0, TArray(TInt32))),
      FastIndexedSeq(("x", I32(0)), ("y", NA(TInt32))),
      "val",
      FastIndexedSeq(Ref("val", TInt32) + Ref("x", TInt32), Coalesce(FastSeq(Ref("y", TInt32), Ref("val", TInt32)))),
      MakeStruct(FastSeq(("x", Ref("x", TInt32)), ("y", Ref("y", TInt32))))
    )

    assertEvalsTo(af, FastIndexedSeq((FastIndexedSeq(1, 2, 3), TArray(TInt32))), Row(6, 1))
  }

  @Test def testArrayScan() {
    implicit val execStrats = ExecStrategy.javaOnly

    def scan(array: IR, zero: IR, f: (IR, IR) => IR): IR =
      ToArray(StreamScan(array, zero, "_accum", "_elt", f(Ref("_accum", zero.typ), Ref("_elt", zero.typ))))

    assertEvalsTo(scan(StreamRange(1, 4, 1), NA(TBoolean), (accum, elt) => IsNA(accum)), FastIndexedSeq(null, true, false, false))
    assertEvalsTo(scan(TestUtils.IRStream(1, 2, 3), 0, (accum, elt) => accum + elt), FastIndexedSeq(0, 1, 3, 6))
    assertEvalsTo(scan(TestUtils.IRStream(1, 2, 3), NA(TInt32), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(TestUtils.IRStream(1, null, 3), NA(TInt32), (accum, elt) => accum + elt), FastIndexedSeq(null, null, null, null))
    assertEvalsTo(scan(NA(TStream(TInt32)), 0, (accum, elt) => accum + elt), null)
    assertEvalsTo(scan(MakeStream(Seq(), TStream(TInt32)), 99, (accum, elt) => accum + elt), FastIndexedSeq(99))
    assertEvalsTo(scan(StreamFlatMap(StreamRange(0, 5, 1), "z", MakeStream(Seq(), TStream(TInt32))), 99, (accum, elt) => accum + elt), FastIndexedSeq(99))
  }

  def makeNDArray(data: Seq[Double], shape: Seq[Long], rowMajor: IR): MakeNDArray = {
    MakeNDArray(MakeArray(data.map(F64), TArray(TFloat64)), MakeTuple.ordered(shape.map(I64)), rowMajor)
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
    implicit val execStrats = ExecStrategy.compileOnly

    assertEvalsTo(NDArrayShape(scalarRowMajor), Row())
    assertEvalsTo(NDArrayShape(vectorRowMajor), Row(2L))
    assertEvalsTo(NDArrayShape(cubeRowMajor), Row(3L, 3L, 3L))
  }

  @Test def testNDArrayRef() {
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

  @Test def testNDArrayReshape() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val v = NDArrayReshape(matrixRowMajor, MakeTuple.ordered(Seq(I64(4))))
    val mat2 = NDArrayReshape(v, MakeTuple.ordered(Seq(I64(2), I64(2))))

    assertEvalsTo(makeNDArrayRef(v, FastIndexedSeq(2)), 3.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastIndexedSeq(1, 0)), 3.0)
    assertEvalsTo(makeNDArrayRef(v, FastIndexedSeq(0)), 1.0)
    assertEvalsTo(makeNDArrayRef(mat2, FastIndexedSeq(0, 0)), 1.0)
  }

  @Test def testNDArrayConcat() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    def nds(ndData: (IndexedSeq[Int], Long, Long)*): IR = {
      MakeArray(ndData.map { case (values, nRows, nCols) =>
        if (values == null) NA(TNDArray(TInt32, Nat(2))) else
          MakeNDArray(Literal(TArray(TInt32), values),
            Literal(TTuple(TInt64, TInt64), Row(nRows, nCols)), True())
      }, TArray(TNDArray(TInt32, Nat(2))))
    }

    val nd1 = (FastIndexedSeq(
      0, 1, 2,
      3, 4, 5), 2L, 3L)

    val rowwise = (FastIndexedSeq(
      6, 7, 8,
      9, 10, 11,
      12, 13, 14), 3L, 3L)

    val colwise = (FastIndexedSeq(
      15, 16,
      17, 18), 2L, 2L)

    val emptyRowwise = (FastIndexedSeq(), 0L, 3L)
    val emptyColwise = (FastIndexedSeq(), 2L, 0L)
    val na = (null, 0L, 0L)

    val rowwiseExpected = FastIndexedSeq(
      FastIndexedSeq(0, 1, 2),
      FastIndexedSeq(3, 4, 5),
      FastIndexedSeq(6, 7, 8),
      FastIndexedSeq(9, 10, 11),
      FastIndexedSeq(12, 13, 14))
    val colwiseExpected = FastIndexedSeq(
      FastIndexedSeq(0, 1, 2, 15, 16),
      FastIndexedSeq(3, 4, 5, 17, 18))

    assertNDEvals(NDArrayConcat(nds(nd1, rowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, rowwise, emptyRowwise), 0), rowwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, emptyRowwise, rowwise), 0), rowwiseExpected)

    assertNDEvals(NDArrayConcat(nds(nd1, colwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, colwise, emptyColwise), 1), colwiseExpected)
    assertNDEvals(NDArrayConcat(nds(nd1, emptyColwise, colwise), 1), colwiseExpected)

    assertNDEvals(NDArrayConcat(nds(nd1, na), 1), null)
    assertNDEvals(NDArrayConcat(nds(na, na), 1), null)
    assertNDEvals(NDArrayConcat(NA(TArray(TNDArray(TInt32, Nat(2)))), 1), null)
  }

  @Test def testNDArrayMap() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val data = 0 until 10
    val shape = FastSeq(2L, 5L)
    val nDim = 2

    val positives = makeNDArray(data.map(_.toDouble), shape, True())
    val negatives = NDArrayMap(positives, "e", ApplyUnaryPrimOp(Negate(), Ref("e", TFloat64)))
    assertEvalsTo(makeNDArrayRef(positives, FastSeq(1L, 0L)), 5.0)
    assertEvalsTo(makeNDArrayRef(negatives, FastSeq(1L, 0L)), -5.0)

    val trues = MakeNDArray(MakeArray(data.map(_ => True()), TArray(TBoolean)), MakeTuple.ordered(shape.map(I64)), True())
    val falses = NDArrayMap(trues, "e", ApplyUnaryPrimOp(Bang(), Ref("e", TBoolean)))
    assertEvalsTo(makeNDArrayRef(trues, FastSeq(1L, 0L)), true)
    assertEvalsTo(makeNDArrayRef(falses, FastSeq(1L, 0L)), false)

    val bools = MakeNDArray(MakeArray(data.map(i => if (i % 2 == 0) True() else False()), TArray(TBoolean)),
      MakeTuple.ordered(shape.map(I64)), False())
    val boolsToBinary = NDArrayMap(bools, "e", If(Ref("e", TBoolean), I64(1L), I64(0L)))
    val one = makeNDArrayRef(boolsToBinary, FastSeq(0L, 0L))
    val zero = makeNDArrayRef(boolsToBinary, FastSeq(1L, 1L))
    assertEvalsTo(one, 1L)
    assertEvalsTo(zero, 0L)
  }

  @Test def testNDArrayMap2() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val shape = MakeTuple.ordered(FastSeq(2L, 2L).map(I64))
    val numbers = MakeNDArray(MakeArray((0 until 4).map { i => F64(i.toDouble) }, TArray(TFloat64)), shape, True())
    val bools = MakeNDArray(MakeArray(Seq(True(), False(), False(), True()), TArray(TBoolean)), shape, True())

    val actual = NDArrayMap2(numbers, bools, "n", "b",
      ApplyBinaryPrimOp(Add(), Ref("n", TFloat64), If(Ref("b", TBoolean), F64(10), F64(20))))
    val ten = makeNDArrayRef(actual, FastSeq(0L, 0L))
    val twentyTwo = makeNDArrayRef(actual, FastSeq(1L, 0L))
    assertEvalsTo(ten, 10.0)
    assertEvalsTo(twentyTwo, 22.0)
  }

  @Test def testNDArrayReindex() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

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
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    val scalarWithMatrix = NDArrayMap2(
      NDArrayReindex(scalarRowMajor, FastIndexedSeq(1, 0)),
      matrixRowMajor,
      "s", "m",
      ApplyBinaryPrimOp(Add(), Ref("s", TFloat64), Ref("m", TFloat64)))

    val topLeft = makeNDArrayRef(scalarWithMatrix, FastIndexedSeq(0, 0))
    assertEvalsTo(topLeft, 4.0)

    val vectorWithMatrix = NDArrayMap2(
      NDArrayReindex(vectorRowMajor, FastIndexedSeq(1, 0)),
      matrixRowMajor,
      "v", "m",
      ApplyBinaryPrimOp(Add(), Ref("v", TFloat64), Ref("m", TFloat64)))

    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(0, 1)), 1.0)
    assertEvalsTo(makeNDArrayRef(vectorWithMatrix, FastIndexedSeq(1, 0)), 4.0)

    val colVector = makeNDArray(FastIndexedSeq(1.0, -1.0), FastIndexedSeq(2, 1), True())
    val colVectorWithMatrix = NDArrayMap2(colVector, matrixRowMajor, "v", "m",
      ApplyBinaryPrimOp(Add(), Ref("v", TFloat64), Ref("m", TFloat64)))

    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(0, 0)), 2.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(0, 1)), 3.0)
    assertEvalsTo(makeNDArrayRef(colVectorWithMatrix, FastIndexedSeq(1, 0)), 2.0)
  }

  @Test(enabled = false) def testNDArrayAgg() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

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
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

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

  @Test def testNDArrayInv() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly
    val matrixRowMajor = makeNDArray(FastSeq(1.5, 2.0, 4.0, 5.0), FastSeq(2, 2), True())
    val inv = NDArrayInv(matrixRowMajor)
    val expectedInv = FastSeq(FastSeq(-10.0, 4.0), FastSeq(8.0, -3.0))
    assertNDEvals(inv, expectedInv)
  }

  @Test def testNDArraySlice() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

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

  @Test def testNDArrayFilter() {
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(NA(TArray(TInt64)), NA(TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(1.0, 2.0),
        FastIndexedSeq(3.0, 4.0)))

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(
        MakeArray(FastIndexedSeq(I64(0), I64(1)), TArray(TInt64)),
        MakeArray(FastIndexedSeq(I64(0), I64(1)), TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(1.0, 2.0),
        FastIndexedSeq(3.0, 4.0)))

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(
        MakeArray(FastIndexedSeq(I64(1), I64(0)), TArray(TInt64)),
        MakeArray(FastIndexedSeq(I64(1), I64(0)), TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(4.0, 3.0),
        FastIndexedSeq(2.0, 1.0)))

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(
        MakeArray(FastIndexedSeq(I64(0)), TArray(TInt64)), NA(TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(1.0, 2.0)))

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(
        NA(TArray(TInt64)), MakeArray(FastIndexedSeq(I64(0)), TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(1.0),
        FastIndexedSeq(3.0)))

    assertNDEvals(
      NDArrayFilter(matrixRowMajor, FastIndexedSeq(
        MakeArray(FastIndexedSeq(I64(1)), TArray(TInt64)),
        MakeArray(FastIndexedSeq(I64(1)), TArray(TInt64)))),
      FastIndexedSeq(FastIndexedSeq(4.0)))
  }

  private def join(left: IR, right: IR, lKeys: IndexedSeq[String], rKeys: IndexedSeq[String], rightDistinct: Boolean, joinType: String): IR = {
    val joinF = { (l: IR, r: IR) =>
      def getL(field: String): IR = GetField(Ref("_left", l.typ), field)
      def getR(field: String): IR = GetField(Ref("_right", r.typ), field)
      Let("_right", r,
          Let("_left", l,
              MakeStruct(
                (lKeys, rKeys).zipped.map { case (lk, rk) => lk -> Coalesce(Seq(getL(lk), getR(rk))) }
                  ++ coerce[TStruct](l.typ).fields.filter(f => !lKeys.contains(f.name)).map { f =>
                  f.name -> GetField(Ref("_left", l.typ), f.name)
                } ++ coerce[TStruct](r.typ).fields.filter(f => !rKeys.contains(f.name)).map { f =>
                  f.name -> GetField(Ref("_right", r.typ), f.name)
                })))
    }
    val mkStream = if (rightDistinct) StreamJoinRightDistinct.apply _ else StreamJoin.apply _
    ToArray(mkStream(left, right, lKeys, rKeys, "_l", "_r",
                     joinF(Ref("_l", coerce[TStream](left.typ).elementType), Ref("_r", coerce[TStream](right.typ).elementType)),
                     joinType))
  }

  @Test def testStreamZipJoin() {
    def eltType = TStruct("k1" -> TInt32, "k2" -> TString, "idx" -> TInt32)
    def makeStream(a: Seq[Integer]): IR = {
      if (a == null)
        NA(TStream(eltType))
      else
        MakeStream(
          a.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "k1" -> (if (n == null) NA(TInt32) else I32(n)),
              "k2" -> Str("x"),
              "idx" -> I32(idx)))},
          TStream(eltType))
    }

    def zipJoin(as: IndexedSeq[Seq[Integer]], key: Int): IR = {
      val streams = as.map(makeStream)
      val keyRef = Ref(genUID(), TStruct(FastIndexedSeq("k1", "k2").take(key).map(k => k -> eltType.fieldType(k)): _*))
      val valsRef = Ref(genUID(), TArray(eltType))
      ToArray(StreamZipJoin(streams, FastIndexedSeq("k1", "k2").take(key), keyRef.name, valsRef.name, InsertFields(keyRef, FastSeq("vals" -> valsRef))))
    }

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](0, 1, null), null), 1),
      null)

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](0, 1, null), Array[Integer](1, 2, null)), 1),
      FastIndexedSeq(
        Row(0, FastIndexedSeq(Row(0, "x", 0), null)),
        Row(1, FastIndexedSeq(Row(1, "x", 1), Row(1, "x", 0))),
        Row(2, FastIndexedSeq(null, Row(2, "x", 1))),
        Row(null, FastIndexedSeq(Row(null, "x", 2), null)),
        Row(null, FastIndexedSeq(null, Row(null, "x", 2)))))

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](0, 1), Array[Integer](1, 2), Array[Integer](0, 2)), 1),
      FastIndexedSeq(
        Row(0, FastIndexedSeq(Row(0, "x", 0), null, Row(0, "x", 0))),
        Row(1, FastIndexedSeq(Row(1, "x", 1), Row(1, "x", 0), null)),
        Row(2, FastIndexedSeq(null, Row(2, "x", 1), Row(2, "x", 1)))))

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](0, 1), Array[Integer](), Array[Integer](0, 2)), 1),
      FastIndexedSeq(
        Row(0, FastIndexedSeq(Row(0, "x", 0), null, Row(0, "x", 0))),
        Row(1, FastIndexedSeq(Row(1, "x", 1), null, null)),
        Row(2, FastIndexedSeq(null, null, Row(2, "x", 1)))))

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](), Array[Integer]()), 1),
      FastIndexedSeq())

    assertEvalsTo(
      zipJoin(FastIndexedSeq(Array[Integer](0, 1)), 1),
      FastIndexedSeq(
        Row(0, FastIndexedSeq(Row(0, "x", 0))),
        Row(1, FastIndexedSeq(Row(1, "x", 1)))))
  }

  @Test def testStreamMultiMerge() {
    def eltType = TStruct("k1" -> TInt32, "k2" -> TString, "idx" -> TInt32)
    def makeStream(a: Seq[Integer]): IR = {
      if (a == null)
        NA(TStream(eltType))
      else
        MakeStream(
          a.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastSeq(
              "k1" -> (if (n == null) NA(TInt32) else I32(n)),
              "k2" -> Str("x"),
              "idx" -> I32(idx)))},
          TStream(eltType))
    }

    def merge(as: IndexedSeq[Seq[Integer]], key: Int): IR = {
      val streams = as.map(makeStream)
      ToArray(StreamMultiMerge(streams, FastIndexedSeq("k1", "k2").take(key)))
    }

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](0, 1, null, null), null), 1),
      null)

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](0, 1, null, null), Array[Integer](1, 2, null, null)), 1),
      FastIndexedSeq(
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(1, "x", 0),
        Row(2, "x", 1),
        Row(null, "x", 2),
        Row(null, "x", 3),
        Row(null, "x", 2),
        Row(null, "x", 3)))

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](0, 1), Array[Integer](1, 2), Array[Integer](0, 2)), 1),
      FastIndexedSeq(
        Row(0, "x", 0),
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(1, "x", 0),
        Row(2, "x", 1),
        Row(2, "x", 1)))

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](0, 1), Array[Integer](), Array[Integer](0, 2)), 1),
      FastIndexedSeq(
        Row(0, "x", 0),
        Row(0, "x", 0),
        Row(1, "x", 1),
        Row(2, "x", 1)))

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](), Array[Integer]()), 1),
      FastIndexedSeq())

    assertEvalsTo(
      merge(FastIndexedSeq(Array[Integer](0, 1)), 1),
      FastIndexedSeq(
        Row(0, "x", 0),
        Row(1, "x", 1)))
  }

  @Test def testJoinRightDistinct() {
    implicit val execStrats = ExecStrategy.javaOnly

    def joinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], joinType: String): IR = {
      join(
        MakeStream.unify(left.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("lk1" -> (if (n == null) NA(TInt32) else I32(n)), "lk2" -> Str("x"), "a" -> I64(idx))) }),
        MakeStream.unify(right.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("b" -> I32(idx), "rk2" -> Str("x"), "rk1" -> (if (n == null) NA(TInt32) else I32(n)), "c" -> Str("foo"))) }),
        FastIndexedSeq("lk1", "lk2"),
        FastIndexedSeq("rk1", "rk2"),
        rightDistinct = true,
        joinType)
    }
    def leftJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "left")
    def outerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "outer")

    assertEvalsTo(
      join(
        NA(TStream(TStruct("k1" -> TInt32, "k2" -> TString, "a" -> TInt64))),
        MakeStream.unify(Seq(MakeStruct(FastIndexedSeq("b" -> I32(0), "k2" -> Str("x"), "k1" -> I32(3), "c" -> Str("foo"))))),
        FastIndexedSeq("k1", "k2"),
        FastIndexedSeq("k1", "k2"),
        true,
        "left"),
      null)

    assertEvalsTo(
      join(
        MakeStream.unify(Seq(MakeStruct(FastIndexedSeq("k1" -> I32(0), "k2" -> Str("x"), "a" -> I64(3))))),
        NA(TStream(TStruct("b" -> TInt32, "k2" -> TString, "k1" -> TInt32, "c" -> TString))),
        FastIndexedSeq("k1", "k2"),
        FastIndexedSeq("k1", "k2"),
        true,
        "left"),
      null)

    assertEvalsTo(leftJoinRows(Array[Integer](0, null), Array[Integer](1, null)), FastIndexedSeq(
      Row(0, "x", 0L, null, null),
      Row(null, "x", 1L, null, null)))

    assertEvalsTo(outerJoinRows(Array[Integer](0, null), Array[Integer](1, null)), FastIndexedSeq(
      Row(0, "x", 0L, null, null),
      Row(1, "x", null, 0, "foo"),
      Row(null, "x", 1L, null, null),
      Row(null, "x", null, 1, "foo")))

    assertEvalsTo(leftJoinRows(Array[Integer](0, 1, 2), Array[Integer](1)), FastIndexedSeq(
      Row(0, "x", 0L, null, null),
      Row(1, "x", 1L, 0, "foo"),
      Row(2, "x", 2L, null, null)))

    assertEvalsTo(leftJoinRows(Array[Integer](0, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)), FastIndexedSeq(
      Row(0, "x", 0L, 1, "foo"),
      Row(1, "x", 1L, 3, "foo"),
      Row(2, "x", 2L, 5, "foo")))

    assertEvalsTo(leftJoinRows(Array[Integer](0, 1, 1, 2), Array[Integer](-1, 0, 0, 1, 1, 2, 2, 3)), FastIndexedSeq(
      Row(0, "x", 0L, 1, "foo"),
      Row(1, "x", 1L, 3, "foo"),
      Row(1, "x", 2L, 3, "foo"),
      Row(2, "x", 3L, 5, "foo")))
  }

  @Test def testStreamJoin() {
    implicit val execStrats = ExecStrategy.javaOnly

    def joinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], joinType: String): IR = {
      join(
        MakeStream.unify(left.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("lk" -> (if (n == null) NA(TInt32) else I32(n)), "l" -> I32(idx))) }),
        MakeStream.unify(right.zipWithIndex.map { case (n, idx) => MakeStruct(FastIndexedSeq("rk" -> (if (n == null) NA(TInt32) else I32(n)), "r" -> I32(idx))) }),
        FastIndexedSeq("lk"),
        FastIndexedSeq("rk"),
        false,
        joinType)
    }
    def leftJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "left")
    def outerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "outer")
    def innerJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "inner")
    def rightJoinRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer]): IR =
      joinRows(left, right, "right")

    assertEvalsTo(leftJoinRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null)), FastIndexedSeq(
      Row(1, 0, 2),
      Row(1, 0, 3),
      Row(1, 1, 2),
      Row(1, 1, 3),
      Row(2, 2, null),
      Row(2, 3, null),
      Row(null, 4, null),
      Row(null, 5, null)))

    assertEvalsTo(outerJoinRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null)), FastIndexedSeq(
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
      Row(null, null, 7)))

    assertEvalsTo(innerJoinRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null)), FastIndexedSeq(
      Row(1, 0, 2),
      Row(1, 0, 3),
      Row(1, 1, 2),
      Row(1, 1, 3)))

    assertEvalsTo(rightJoinRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null)), FastIndexedSeq(
      Row(0, null, 0),
      Row(0, null, 1),
      Row(1, 0, 2),
      Row(1, 0, 3),
      Row(1, 1, 2),
      Row(1, 1, 3),
      Row(3, null, 4),
      Row(3, null, 5),
      Row(null, null, 6),
      Row(null, null, 7)))
  }

  @Test def testStreamMerge() {
    implicit val execStrats = ExecStrategy.javaOnly

    def mergeRows(left: IndexedSeq[Integer], right: IndexedSeq[Integer], key: Int): IR = {
      val typ = TStream(TStruct("k" -> TInt32, "sign" -> TInt32, "idx" -> TInt32))
      ToArray(StreamMerge(
        if (left == null)
          NA(typ)
        else
          MakeStream(left.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastIndexedSeq(
              "k" -> (if (n == null) NA(TInt32) else I32(n)),
              "sign" -> I32(1),
              "idx" -> I32(idx)))
          }, typ),
        if (right == null)
          NA(typ)
        else
          MakeStream(right.zipWithIndex.map { case (n, idx) =>
            MakeStruct(FastIndexedSeq(
              "k" -> (if (n == null) NA(TInt32) else I32(n)),
              "sign" -> I32(-1),
              "idx" -> I32(idx)))
          }, typ),
        FastIndexedSeq("k", "sign").take(key)))
    }

    assertEvalsTo(mergeRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null), 1), FastIndexedSeq(
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
      Row(null, -1, 7)))

    // right stream ends first
    assertEvalsTo(mergeRows(Array[Integer](1, 1, 2, 2), Array[Integer](0, 0, 1, 1), 1), FastIndexedSeq(
      Row(0, -1, 0),
      Row(0, -1, 1),
      Row(1, 1, 0),
      Row(1, 1, 1),
      Row(1, -1, 2),
      Row(1, -1, 3),
      Row(2, 1, 2),
      Row(2, 1, 3)))

    // compare on two key fields
    assertEvalsTo(mergeRows(Array[Integer](1, 1, 2, 2, null, null), Array[Integer](0, 0, 1, 1, 3, 3, null, null), 2), FastIndexedSeq(
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
      Row(null, -1, 7)))

    // right stream empty
    assertEvalsTo(mergeRows(Array[Integer](1, 2, null), Array[Integer](), 1), FastIndexedSeq(
      Row(1, 1, 0),
      Row(2, 1, 1),
      Row(null, 1, 2)))

    // left stream empty
    assertEvalsTo(mergeRows(Array[Integer](), Array[Integer](1, 2, null), 1), FastIndexedSeq(
      Row(1, -1, 0),
      Row(2, -1, 1),
      Row(null, -1, 2)))

    // one stream missing
    assertEvalsTo(mergeRows(null, Array[Integer](1, 2, null), 1), null)
    assertEvalsTo(mergeRows(Array[Integer](1, 2, null), null, 1), null)
  }

  @Test def testDie() {
    assertFatal(Die("mumblefoo", TFloat64), "mble")
    assertFatal(Die(NA(TString), TFloat64), "message missing")
  }

  @Test def testDieInferPType() {
    assertPType(Die("mumblefoo", TFloat64), PFloat64(true))
    assertPType(Die("mumblefoo", TArray(TFloat64)), PCanonicalArray(PFloat64(true), true))
  }

  @Test def testStreamRange() {
    def assertEquals(start: Integer, stop: Integer, step: Integer, expected: IndexedSeq[Int]) {
      assertEvalsTo(ToArray(StreamRange(In(0, TInt32), In(1, TInt32), In(2, TInt32))),
        args = FastIndexedSeq(start -> TInt32, stop -> TInt32, step -> TInt32),
        expected = expected)
    }
    assertEquals(0, 5, null, null)
    assertEquals(0, null, 1, null)
    assertEquals(null, 5, 1, null)

    assertFatal(ToArray(StreamRange(I32(0), I32(5), I32(0))), "step size")

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
    implicit val execStrats = ExecStrategy.compileOnly

    val sumSig = AggSignature(Sum(), Seq(), Seq(TInt64))
    assertEvalsTo(
      StreamAgg(
        StreamMap(StreamRange(I32(0), I32(4), I32(1)), "x", Cast(Ref("x", TInt32), TInt64)),
        "x",
        ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq(Ref("x", TInt64)), sumSig)),
      6L)
  }

  @Test def testArrayAggContexts() {
    implicit val execStrats = ExecStrategy.compileOnly

    val ir = Let(
      "x",
      In(0, TInt32) * In(0, TInt32), // multiply to prevent forwarding
      StreamAgg(
        StreamRange(I32(0), I32(10), I32(1)),
        "elt",
        AggLet("y",
          Cast(Ref("x", TInt32) * Ref("x", TInt32) * Ref("elt", TInt32), TInt64), // different type to trigger validation errors
          invoke("append", TArray(TArray(TInt32)),
            ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(
              MakeArray(FastSeq(
                Ref("x", TInt32),
                Ref("elt", TInt32),
                Cast(Ref("y", TInt64), TInt32),
                Cast(Ref("y", TInt64), TInt32)), // reference y twice to prevent forwarding
                TArray(TInt32))),
              AggSignature(Collect(), FastIndexedSeq(), FastIndexedSeq(TArray(TInt32)))),
            MakeArray(FastSeq(Ref("x", TInt32)), TArray(TInt32))),
          isScan = false)))

    assertEvalsTo(ir, FastIndexedSeq(1 -> TInt32),
      (0 until 10).map(i => FastIndexedSeq(1, i, i, i)) ++ FastIndexedSeq(FastIndexedSeq(1)))
  }

  @Test def testStreamAggScan() {
    implicit val execStrats = ExecStrategy.compileOnly

    val eltType = TStruct("x" -> TCall, "y" -> TInt32)

    val ir = (StreamAggScan(ToStream(In(0, TArray(eltType))),
      "foo",
      GetField(Ref("foo", eltType), "y") +
        GetField(ApplyScanOp(
          FastIndexedSeq(I32(2)),
          FastIndexedSeq(GetField(Ref("foo", eltType), "x")),
          AggSignature(CallStats(), FastIndexedSeq(TInt32), FastIndexedSeq(TCall))
        ), "AN")))

    val input = FastIndexedSeq(
      Row(null, 1),
      Row(Call2(0, 0), 2),
      Row(Call2(0, 1), 3),
      Row(Call2(1, 1), 4),
      null,
      Row(null, 5)) -> TArray(eltType)

    assertEvalsTo(ToArray(ir),
      args = FastIndexedSeq(input),
      expected = FastIndexedSeq(1 + 0, 2 + 0, 3 + 2, 4 + 4, null, 5 + 6))

    assertEvalsTo(StreamLen(ir), args=FastIndexedSeq(input), 6)
  }

  @Test def testInsertFields() {
    implicit val execStrats = ExecStrategy.javaOnly

    val s = TStruct("a" -> TInt64, "b" -> TString)
    val emptyStruct = MakeStruct(Seq("a" -> NA(TInt64), "b" -> NA(TString)))

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
        Seq("c" -> NA(TFloat64))),
      Row(null, null, null))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64), "b" -> Str("abc"))),
        Seq()),
      Row(null, "abc"))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64), "b" -> Str("abc"))),
        Seq("a" -> I64(5))),
      Row(5L, "abc"))

    assertEvalsTo(
      InsertFields(
        MakeStruct(Seq("a" -> NA(TInt64), "b" -> Str("abc"))),
        Seq("c" -> F64(3.2))),
      Row(null, "abc", 3.2))

    assertEvalsTo(
      InsertFields(NA(TStruct("a" -> TInt32)), Seq("foo" -> I32(5))),
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
        NA(TStruct("foo" -> TInt32, "bar" -> TFloat64)),
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

    val s = MakeStruct(Seq("a" -> NA(TInt64), "b" -> Str("abc")))
    val na = NA(TStruct("a" -> TInt64, "b" -> TString))

    assertEvalsTo(GetField(s, "a"), null)
    assertEvalsTo(GetField(s, "b"), "abc")
    assertEvalsTo(GetField(na, "a"), null)
  }

  @Test def testLiteral() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.JvmCompile)
    val poopEmoji = new String(Array[Char](0xD83D, 0xDCA9))
    val types = Array(
      TTuple(TInt32, TString, TArray(TInt32)),
      TArray(TString),
      TDict(TInt32, TString)
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
    assertEvalsTo(ApplyComparisonOp(EQ(TArray(TInt32)),
      ToArray(StreamMap(ToStream(Literal(TArray(TFloat64), FastIndexedSeq(1.0, 2.0))), "elt", Cast(Ref("elt", TFloat64), TInt32))),
      Literal(TArray(TInt32), FastIndexedSeq(1, 2))), true)
  }

  @Test def testTableCount() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
    assertEvalsTo(TableCount(TableRange(0, 4)), 0L)
    assertEvalsTo(TableCount(TableRange(7, 4)), 7L)
  }

  @Test def testTableGetGlobals() {
    implicit val execStrats = ExecStrategy.interpretOnly
    assertEvalsTo(TableGetGlobals(TableMapGlobals(TableRange(0, 1), Literal(TStruct("a" -> TInt32), Row(1)))), Row(1))
  }

  @Test def testTableAggregate() {
    implicit val execStrats = ExecStrategy.allRelational

    val table = TableRange(3, 2)
    val countSig = AggSignature(Count(), Seq(), Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq.empty, countSig)
    assertEvalsTo(TableAggregate(table, MakeStruct(Seq("foo" -> count))), Row(3L))
  }

  @Test def testMatrixAggregate() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val matrix = MatrixIR.range(5, 5, None)
    val countSig = AggSignature(Count(), Seq(), Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq.empty, countSig)
    assertEvalsTo(MatrixAggregate(matrix, MakeStruct(Seq("foo" -> count))), Row(25L))
  }

  @Test def testGroupByKey() {
    implicit val execStrats = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.JvmCompile, ExecStrategy.JvmCompileUnoptimized)

    def tuple(k: String, v: Int): IR = MakeTuple.ordered(Seq(Str(k), I32(v)))

    def groupby(tuples: IR*): IR = GroupByKey(MakeStream(tuples, TStream(TTuple(TString, TInt32))))

    val collection1 = groupby(tuple("foo", 0), tuple("bar", 4), tuple("foo", -1), tuple("bar", 0), tuple("foo", 10), tuple("", 0))
    assertEvalsTo(collection1, Map("" -> FastIndexedSeq(0), "bar" -> FastIndexedSeq(4, 0), "foo" -> FastIndexedSeq(0, -1, 10)))
  }

  @DataProvider(name = "compareDifferentTypes")
  def compareDifferentTypesData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(0.0, 0.0), TArray(TFloat64), TArray(TFloat64)),
    Array(Set(0, 1), TSet(TInt32), TSet(TInt32)),
    Array(Map(0L -> 5, 3L -> 20), TDict(TInt64, TInt32), TDict(TInt64, TInt32)),
    Array(Interval(1, 2, includesStart = false, includesEnd = true), TInterval(TInt32), TInterval(TInt32)),
    Array(Row("foo", 0.0), TStruct("a" -> TString, "b" -> TFloat64), TStruct("a" -> TString, "b" -> TFloat64)),
    Array(Row("foo", 0.0), TTuple(TString, TFloat64), TTuple(TString, TFloat64)),
    Array(Row(FastIndexedSeq("foo"), 0.0), TTuple(TArray(TString), TFloat64), TTuple(TArray(TString), TFloat64))
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
    withExecuteContext() { ctx =>
      valueIRs(ctx)
    }
  }

  def valueIRs(ctx: ExecuteContext): Array[Array[IR]] = {
    val fs = ctx.fs

    IndexBgen(ctx, Array("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))

    val b = True()
    val c = Ref("c", TBoolean)
    val i = I32(5)
    val j = I32(7)
    val str = Str("Hail")
    val a = Ref("a", TArray(TInt32))
    val st = Ref("st", TStream(TInt32))
    val aa = Ref("aa", TArray(TArray(TInt32)))
    val sta = Ref("sta", TStream(TArray(TInt32)))
    val da = Ref("da", TArray(TTuple(TInt32, TString)))
    val std = Ref("std", TStream(TTuple(TInt32, TString)))
    val v = Ref("v", TInt32)
    val s = Ref("s", TStruct("x" -> TInt32, "y" -> TInt64, "z" -> TFloat64))
    val t = Ref("t", TTuple(TInt32, TInt64, TFloat64))
    val l = Ref("l", TInt32)
    val r = Ref("r", TInt32)

    val call = Ref("call", TCall)

    val collectSig = AggSignature(Collect(), Seq(), Seq(TInt32))
    val pCollectSig = PhysicalAggSig(Collect(), CollectStateSig(PInt32()))

    val sumSig = AggSignature(Sum(), Seq(), Seq(TInt64))
    val pSumSig = PhysicalAggSig(Sum(), TypedStateSig(PInt64(true)))

    val callStatsSig = AggSignature(CallStats(), Seq(TInt32), Seq(TCall))
    val pCallStatsSig = PhysicalAggSig(CallStats(), CallStatsStateSig())

    val takeBySig = AggSignature(TakeBy(), Seq(TInt32), Seq(TFloat64, TInt32))

    val countSig = AggSignature(Count(), Seq(), Seq())
    val count = ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq.empty, countSig)

    val groupSignature = GroupedAggSig(PInt32(true), FastSeq(pSumSig))

    val table = TableRange(100, 10)

    val mt = MatrixIR.range(20, 2, Some(3))
    val vcf = is.hail.TestUtils.importVCF(ctx, "src/test/resources/sample.vcf")

    val bgenReader = MatrixBGENReader(ctx, FastIndexedSeq("src/test/resources/example.8bits.bgen"), None, Map.empty[String, String], None, None, None)
    val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

    val blockMatrix = BlockMatrixRead(BlockMatrixNativeReader(fs, "src/test/resources/blockmatrix_example/0"))
    val blockMatrixWriter = BlockMatrixNativeWriter("/path/to/file.bm", false, false, false)
    val blockMatrixMultiWriter = BlockMatrixBinaryMultiWriter("/path/to/prefix", false)
    val nd = MakeNDArray(MakeArray(FastSeq(I32(-1), I32(1)), TArray(TInt32)),
      MakeTuple.ordered(FastSeq(I64(1), I64(2))),
      True())

    val irs = Array(
      i, I64(5), F32(3.14f), F64(3.14), str, True(), False(), Void(),
      UUID4(),
      Cast(i, TFloat64),
      CastRename(NA(TStruct("a" -> TInt32)), TStruct("b" -> TInt32)),
      NA(TInt32), IsNA(i),
      If(b, i, j),
      Coalesce(FastSeq(In(0, TInt32), I32(1))),
      Let("v", i, v),
      AggLet("v", i, v, false),
      Ref("x", TInt32),
      ApplyBinaryPrimOp(Add(), i, j),
      ApplyUnaryPrimOp(Negate(), i),
      ApplyComparisonOp(EQ(TInt32), i, j),
      MakeArray(FastSeq(i, NA(TInt32), I32(-3)), TArray(TInt32)),
      MakeStream(FastSeq(i, NA(TInt32), I32(-3)), TStream(TInt32)),
      nd,
      NDArrayReshape(nd, MakeTuple.ordered(Seq(I64(4)))),
      NDArrayConcat(MakeArray(FastSeq(nd, nd), TArray(nd.typ)), 0),
      NDArrayRef(nd, FastSeq(I64(1), I64(2))),
      NDArrayMap(nd, "v", ApplyUnaryPrimOp(Negate(), v)),
      NDArrayMap2(nd, nd, "l", "r", ApplyBinaryPrimOp(Add(), l, r)),
      NDArrayReindex(nd, FastIndexedSeq(0, 1)),
      NDArrayAgg(nd, FastIndexedSeq(0)),
      NDArrayWrite(nd, Str("/path/to/ndarray")),
      NDArrayMatMul(nd, nd),
      NDArraySlice(nd, MakeTuple.ordered(FastSeq(MakeTuple.ordered(FastSeq(F64(0), F64(2), F64(1))),
                                         MakeTuple.ordered(FastSeq(F64(0), F64(2), F64(1)))))),
      NDArrayFilter(nd, FastIndexedSeq(NA(TArray(TInt64)), NA(TArray(TInt64)))),
      ArrayRef(a, i),
      ArrayLen(a),
      StreamLen(st),
      StreamRange(I32(0), I32(5), I32(1)),
      StreamRange(I32(0), I32(5), I32(1)),
      ArraySort(st, b),
      ToSet(st),
      ToDict(std),
      ToArray(st),
      CastToArray(NA(TSet(TInt32))),
      ToStream(a),
      LowerBoundOnOrderedCollection(a, i, onKey = true),
      GroupByKey(da),
      StreamTake(st, I32(10)),
      StreamDrop(st, I32(10)),
      StreamMap(st, "v", v),
      StreamMerge(
        StreamMap(StreamRange(0, 2, 1), "x", MakeStruct(FastSeq("x" -> Ref("x", TInt32)))),
        StreamMap(StreamRange(0, 3, 1), "x", MakeStruct(FastSeq("x" -> Ref("x", TInt32)))),
        FastSeq("x")),
      StreamZip(FastIndexedSeq(st, st), FastIndexedSeq("foo", "bar"), True(), ArrayZipBehavior.TakeMinLength),
      StreamFilter(st, "v", b),
      StreamFlatMap(sta, "v", ToStream(a)),
      StreamFold(st, I32(0), "x", "v", v),
      StreamFold2(StreamFold(st, I32(0), "x", "v", v)),
      StreamScan(st, I32(0), "x", "v", v),
      StreamJoinRightDistinct(
        StreamMap(StreamRange(0, 2, 1), "x", MakeStruct(FastSeq("x" -> Ref("x", TInt32)))),
        StreamMap(StreamRange(0, 3, 1), "x", MakeStruct(FastSeq("x" -> Ref("x", TInt32)))),
        FastIndexedSeq("x"), FastIndexedSeq("x"), "l", "r", I32(1), "left"),
      StreamFor(st, "v", Void()),
      StreamAgg(st, "x", ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq(Cast(Ref("x", TInt32), TInt64)), sumSig)),
      StreamAggScan(st, "x", ApplyScanOp(FastIndexedSeq.empty, FastIndexedSeq(Cast(Ref("x", TInt32), TInt64)), sumSig)),
      RunAgg(Begin(FastSeq(
        InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq(InitOp(0, FastSeq(), pSumSig)))), groupSignature),
        SeqOp(0, FastSeq(I32(1), SeqOp(0, FastSeq(), pSumSig)), groupSignature))),
        AggStateValue(0, groupSignature.state), FastIndexedSeq(groupSignature.state)),
      RunAggScan(StreamRange(I32(0), I32(1), I32(1)),
        "foo",
        InitOp(0, FastIndexedSeq(Begin(FastIndexedSeq(InitOp(0, FastSeq(), pSumSig)))), groupSignature),
        SeqOp(0, FastSeq(Ref("foo", TInt32), SeqOp(0, FastSeq(), pSumSig)), groupSignature),
        AggStateValue(0, groupSignature.state),
        FastIndexedSeq(groupSignature.state)),
      AggFilter(True(), I32(0), false),
      AggExplode(NA(TStream(TInt32)), "x", I32(0), false),
      AggGroupBy(True(), I32(0), false),
      ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq(I32(0)), collectSig),
      ApplyAggOp(FastIndexedSeq(I32(2)), FastIndexedSeq(call), callStatsSig),
      ApplyAggOp(FastIndexedSeq(I32(10)), FastIndexedSeq(F64(-2.11), I32(4)), takeBySig),
      InitOp(0, FastIndexedSeq(I32(2)), pCallStatsSig),
      SeqOp(0, FastIndexedSeq(i), pCollectSig),
      CombOp(0, 1, pCollectSig),
      ResultOp(0, FastIndexedSeq(pCollectSig)),
      SerializeAggs(0, 0, BufferSpec.default, FastSeq(pCollectSig.state)),
      DeserializeAggs(0, 0, BufferSpec.default, FastSeq(pCollectSig.state)),
      CombOpValue(0, In(0, TBinary), pCollectSig),
      AggStateValue(0, pCollectSig.state),
      InitFromSerializedValue(0, In(0, TBinary), pCollectSig.state),
      Begin(FastIndexedSeq(Void())),
      MakeStruct(FastIndexedSeq("x" -> i)),
      SelectFields(s, FastIndexedSeq("x", "z")),
      InsertFields(s, FastIndexedSeq("x" -> i)),
      InsertFields(s, FastIndexedSeq("* x *" -> i)), // Won't parse as a simple identifier
      GetField(s, "x"),
      MakeTuple(FastIndexedSeq(2 -> i, 4 -> b)),
      GetTupleElement(t, 1),
      In(2, TFloat64),
      Die("mumblefoo", TFloat64),
      invoke("land", TBoolean, b, c), // ApplySpecial
      invoke("toFloat64", TFloat64, i), // Apply
      Literal(TStruct("x" -> TInt32), Row(1)),
      TableCount(table),
      MatrixCount(mt),
      TableGetGlobals(table),
      TableCollect(table),
      TableAggregate(table, MakeStruct(Seq("foo" -> count))),
      TableToValueApply(table, ForceCountTable()),
      MatrixToValueApply(mt, ForceCountMatrixTable()),
      TableWrite(table, TableNativeWriter("/path/to/data.ht")),
      MatrixWrite(mt, MatrixNativeWriter("/path/to/data.mt")),
      MatrixWrite(vcf, MatrixVCFWriter("/path/to/sample.vcf")),
      MatrixWrite(vcf, MatrixPLINKWriter("/path/to/base")),
      MatrixWrite(bgen, MatrixGENWriter("/path/to/base")),
      MatrixMultiWrite(Array(mt, mt), MatrixNativeMultiWriter("/path/to/prefix")),
      TableMultiWrite(Array(table, table), WrappedMatrixNativeMultiWriter(MatrixNativeMultiWriter("/path/to/prefix"), FastIndexedSeq("foo"))),
      MatrixAggregate(mt, MakeStruct(Seq("foo" -> count))),
      BlockMatrixCollect(blockMatrix),
      BlockMatrixWrite(blockMatrix, blockMatrixWriter),
      BlockMatrixMultiWrite(IndexedSeq(blockMatrix, blockMatrix), blockMatrixMultiWriter),
      BlockMatrixWrite(blockMatrix, BlockMatrixPersistWriter("x", "MEMORY_ONLY")),
      UnpersistBlockMatrix(blockMatrix),
      CollectDistributedArray(StreamRange(0, 3, 1), 1, "x", "y", Ref("x", TInt32)),
      ReadPartition(Str("foo"),
        TStruct("foo" -> TInt32),
        PartitionNativeReader(TypedCodecSpec(PCanonicalStruct("foo" -> PInt32(), "bar" -> PCanonicalString()), BufferSpec.default))),
      WritePartition(
        MakeStream(FastSeq(), TStream(TStruct())), NA(TString),
        PartitionNativeWriter(TypedCodecSpec(PType.canonical(TStruct()), BufferSpec.default), "path", None, None)),
      WriteMetadata(
        NA(TStruct("global" -> TString, "partitions" -> TStruct("filePath" -> TString, "partitionCounts" -> TInt64))),
        RelationalWriter("path", overwrite = false, None)),
      ReadValue(Str("foo"), TypedCodecSpec(PCanonicalStruct("foo" -> PInt32(), "bar" -> PCanonicalString()), BufferSpec.default), TStruct("foo" -> TInt32)),
      WriteValue(I32(1), Str("foo"), TypedCodecSpec(PInt32(), BufferSpec.default)),
      LiftMeOut(I32(1)),
      RelationalLet("x", I32(0), I32(0)),
      TailLoop("y", IndexedSeq("x" -> I32(0)), Recur("y", FastSeq(I32(4)), TInt32)),
      {
        val keyFields = FastIndexedSeq(SortField("foo", Ascending))
        val rowType = TStruct("foo" -> TInt32)
        val rowEType = EBaseStruct(FastIndexedSeq(EField("foo", EInt32Required, 0)))
        val keyEType = EBaseStruct(FastIndexedSeq(EField("foo", EInt32Required, 0)))
        val shuffleType = TShuffle(keyFields, rowType, rowEType, keyEType)
        ShuffleWith(keyFields, rowType, rowEType, keyEType,
          "id",
          ShuffleWrite(
            Ref("id", shuffleType),
            MakeArray(MakeStruct(FastSeq(("foo", I32(0)))))),
          Let(
            "garbage",
            ShufflePartitionBounds(
              Ref("id", shuffleType),
              I32(1)),
            ShuffleRead(
              Ref("id", shuffleType),
              ApplySpecial("Interval",
                FastSeq(),
                FastSeq(I32(0), I32(5), True(), False()),
                TInterval(TInt32)))))
      }
      )
    irs.map(x => Array(x))
  }

  @DataProvider(name = "tableIRs")
  def tableIRs(): Array[Array[TableIR]] = {
    withExecuteContext() { ctx =>
      tableIRs(ctx)
    }
  }

  def tableIRs(ctx: ExecuteContext): Array[Array[TableIR]] = {
    try {
      val fs = ctx.fs

      val read = TableIR.read(fs, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
      val mtRead = MatrixIR.read(fs, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")
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
          NA(TStruct.empty), NA(TStruct.empty), Some(1), 2),
        TableJoin(read,
          TableRange(100, 10), "inner", 1),
        TableLeftJoinRightDistinct(read, TableRange(100, 10), "root"),
        TableMultiWayZipJoin(FastIndexedSeq(read, read), " * data * ", "globals"),
        MatrixEntriesTable(mtRead),
        MatrixRowsTable(mtRead),
        TableRepartition(read, 10, RepartitionStrategy.COALESCE),
        TableHead(read, 10),
        TableTail(read, 10),
        TableGroupWithinPartitions(read, "_grouped", 3),
        TableParallelize(
          MakeStruct(FastSeq(
            "rows" -> MakeArray(FastSeq(
            MakeStruct(FastSeq("a" -> NA(TInt32))),
            MakeStruct(FastSeq("a" -> I32(1)))
          ), TArray(TStruct("a" -> TInt32))),
            "global" -> MakeStruct(FastSeq()))), None),
        TableMapRows(TableKeyBy(read, FastIndexedSeq()),
          MakeStruct(FastIndexedSeq(
            "a" -> GetField(Ref("row", read.typ.rowType), "f32"),
            "b" -> F64(-2.11)))),
        TableMapGlobals(read,
          MakeStruct(FastIndexedSeq(
            "foo" -> NA(TArray(TInt32))))),
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
    withExecuteContext() { ctx =>
      matrixIRs(ctx)
    }
  }

  def matrixIRs(ctx: ExecuteContext): Array[Array[MatrixIR]] = {
    try {
      val fs = ctx.fs

      IndexBgen(ctx, Array("src/test/resources/example.8bits.bgen"), rg = Some("GRCh37"), contigRecoding = Map("01" -> "1"))

      val tableRead = TableIR.read(fs, "src/test/resources/backward_compatability/1.0.0/table/0.ht")
      val read = MatrixIR.read(fs, "src/test/resources/backward_compatability/1.0.0/matrix_table/0.hmt")
      val range = MatrixIR.range(3, 7, None)
      val vcf = is.hail.TestUtils.importVCF(ctx, "src/test/resources/sample.vcf")

      val bgenReader = MatrixBGENReader(ctx, FastIndexedSeq("src/test/resources/example.8bits.bgen"), None, Map.empty[String, String], None, None, None)
      val bgen = MatrixRead(bgenReader.fullMatrixType, false, false, bgenReader)

      val range1 = MatrixIR.range(20, 2, Some(3))
      val range2 = MatrixIR.range(20, 2, Some(4))

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

      val collectSig = AggSignature(Collect(), Seq(), Seq(TInt32))
      val collect = ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq(I32(0)), collectSig)

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
        MatrixRowsTail(range1, 3),
        MatrixColsTail(range1, 3),
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
        RelationalLetMatrixTable("x", I32(0), read))

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
    val read = BlockMatrixRead(BlockMatrixNativeReader(fs, "src/test/resources/blockmatrix_example/0"))
    val transpose = BlockMatrixBroadcast(read, FastIndexedSeq(1, 0), FastIndexedSeq(2, 2), 2)
    val dot = BlockMatrixDot(read, transpose)
    val slice = BlockMatrixSlice(read, FastIndexedSeq(FastIndexedSeq(0, 2, 1), FastIndexedSeq(0, 1, 1)))

    val sparsify1 = BlockMatrixSparsify(read, RectangleSparsifier(FastIndexedSeq(FastIndexedSeq(0L, 1L, 5L, 6L))))
    val sparsify2 = BlockMatrixSparsify(read, BandSparsifier(true, -1L, 1L))
    val sparsify3 = BlockMatrixSparsify(read, RowIntervalSparsifier(true, FastIndexedSeq(0L, 1L, 5L, 6L), FastIndexedSeq(5L, 6L, 8L, 9L)))
    val densify = BlockMatrixDensify(read)

    val blockMatrixIRs = Array[BlockMatrixIR](read,
      transpose,
      dot,
      sparsify1,
      sparsify2,
      sparsify3,
      densify,
      RelationalLetBlockMatrix("x", I32(0), read),
      slice,
      BlockMatrixRead(BlockMatrixPersistReader("x"))
    )

    blockMatrixIRs.map(ir => Array(ir))
  }

  @Test def testIRConstruction(): Unit = {
    matrixIRs()
    tableIRs()
    valueIRs()
    blockMatrixIRs()
  }

  @Test(dataProvider = "valueIRs")
  def testValueIRParser(x: IR) {
    val env = IRParserEnvironment(ctx, refMap = Map(
      "c" -> TBoolean,
      "a" -> TArray(TInt32),
      "aa" -> TArray(TArray(TInt32)),
      "da" -> TArray(TTuple(TInt32, TString)),
      "st" -> TStream(TInt32),
      "sta" -> TStream(TArray(TInt32)),
      "std" -> TStream(TTuple(TInt32, TString)),
      "nd" -> TNDArray(TFloat64, Nat(1)),
      "nd2" -> TNDArray(TArray(TString), Nat(1)),
      "v" -> TInt32,
      "l" -> TInt32,
      "r" -> TInt32,
      "s" -> TStruct("x" -> TInt32, "y" -> TInt64, "z" -> TFloat64),
      "t" -> TTuple(TInt32, TInt64, TFloat64),
      "call" -> TCall,
      "x" -> TInt32))

    val s = Pretty(x, elideLiterals = false)

    val x2 = IRParser.parse_value_ir(s, env)

    assert(x2 == x)
  }

  @Test(dataProvider = "tableIRs")
  def testTableIRParser(x: TableIR) {
    val s = Pretty(x, elideLiterals = false)
    val x2 = IRParser.parse_table_ir(ctx, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "matrixIRs")
  def testMatrixIRParser(x: MatrixIR) {
    val s = Pretty(x, elideLiterals = false)
    val x2 = IRParser.parse_matrix_ir(ctx, s)
    assert(x2 == x)
  }

  @Test(dataProvider = "blockMatrixIRs")
  def testBlockMatrixIRParser(x: BlockMatrixIR) {
    val s = Pretty(x, elideLiterals = false)
    val x2 = IRParser.parse_blockmatrix_ir(ctx, s)
    assert(x2 == x)
  }

  @Test def testCachedIR() {
    val cached = Literal(TSet(TInt32), Set(1))
    val s = s"(JavaIR __uid1)"
    val x2 = ExecuteContext.scoped() { ctx =>
      IRParser.parse_value_ir(s, IRParserEnvironment(ctx, refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    }
    assert(x2 eq cached)
  }

  @Test def testCachedTableIR() {
    val cached = TableRange(1, 1)
    val s = s"(JavaTable __uid1)"
    val x2 = ExecuteContext.scoped() { ctx =>
      IRParser.parse_table_ir(s, IRParserEnvironment(ctx, refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    }
    assert(x2 eq cached)
  }

  @Test def testCachedMatrixIR() {
    val cached = MatrixIR.range(3, 7, None)
    val s = s"(JavaMatrix __uid1)"
    val x2 = ExecuteContext.scoped() { ctx =>
      IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    }
    assert(x2 eq cached)
  }

  @Test def testCachedBlockMatrixIR() {
    val cached = new BlockMatrixLiteral(BlockMatrix.fill(3, 7, 1))
    val s = s"(JavaBlockMatrix __uid1)"
    val x2 = ExecuteContext.scoped() { ctx =>
      IRParser.parse_blockmatrix_ir(s, IRParserEnvironment(ctx, refMap = Map.empty, irMap = Map("__uid1" -> cached)))
    }
    assert(x2 eq cached)
  }

  @Test def testContextSavedMatrixIR() {
    val cached = MatrixIR.range(3, 8, None)
    val id = hc.addIrVector(Array(cached))
    val s = s"(JavaMatrixVectorRef $id 0)"
    val x2 = ExecuteContext.scoped() { ctx =>
      IRParser.parse_matrix_ir(s, IRParserEnvironment(ctx, refMap = Map.empty, irMap = Map.empty))
    }
    assert(cached eq x2)

    is.hail.HailContext.pyRemoveIrVector(id)
    assert(hc.irVectors.get(id) eq None)
  }

  @Test def testEvaluations() {
    TestFunctions.registerAll()

    def test(x: IR, i: java.lang.Boolean, expectedEvaluations: Int) {
      val env = Env.empty[(Any, Type)]
      val args = FastIndexedSeq((i, TBoolean))

      IRSuite.globalCounter = 0
      Interpret[Any](ctx, x, env, args, optimize = false)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      Interpret[Any](ctx, x, env, args)
      assert(IRSuite.globalCounter == expectedEvaluations)

      IRSuite.globalCounter = 0
      eval(x, env, args, None)
      assert(IRSuite.globalCounter == expectedEvaluations)
    }

    def i = In(0, TBoolean)

    def st = ApplySeeded("incr_s", FastSeq(True()), 0L, TBoolean)

    def sf = ApplySeeded("incr_s", FastSeq(True()), 0L, TBoolean)

    def sm = ApplySeeded("incr_s", FastSeq(NA(TBoolean)), 0L, TBoolean)

    def vt = ApplySeeded("incr_v", FastSeq(True()), 0L, TBoolean)

    def vf = ApplySeeded("incr_v", FastSeq(True()), 0L, TBoolean)

    def vm = ApplySeeded("incr_v", FastSeq(NA(TBoolean)), 0L, TBoolean)

    // baseline
    test(st, true, 1); test(sf, true, 1); test(sm, true, 1)
    test(vt, true, 1); test(vf, true, 1); test(vm, true, 0)

    // if
    // condition
    test(If(st, i, True()), true, 1)
    test(If(sf, i, True()), true, 1)
    test(If(sm, i, True()), true, 1)

    test(If(vt, i, True()), true, 1)
    test(If(vf, i, True()), true, 1)
    test(If(vm, i, True()), true, 0)

    // consequent
    test(If(i, st, True()), true, 1)
    test(If(i, sf, True()), true, 1)
    test(If(i, sm, True()), true, 1)

    test(If(i, vt, True()), true, 1)
    test(If(i, vf, True()), true, 1)
    test(If(i, vm, True()), true, 0)

    // alternate
    test(If(i, True(), st), false, 1)
    test(If(i, True(), sf), false, 1)
    test(If(i, True(), sm), false, 1)

    test(If(i, True(), vt), false, 1)
    test(If(i, True(), vf), false, 1)
    test(If(i, True(), vm), false, 0)
  }

  @Test def testArrayContinuationDealsWithIfCorrectly() {
    val ir = ToArray(StreamMap(
      If(IsNA(In(0, TBoolean)),
        NA(TStream(TInt32)),
        ToStream(In(1, TArray(TInt32)))),
      "x", Cast(Ref("x", TInt32), TInt64)))

    assertEvalsTo(ir, FastIndexedSeq(true -> TBoolean, FastIndexedSeq(0) -> TArray(TInt32)), FastIndexedSeq(0L))
  }

  @Test def testTableGetGlobalsSimplifyRules() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t1 = TableType(TStruct("a" -> TInt32), FastIndexedSeq("a"), TStruct("g1" -> TInt32, "g2" -> TFloat64))
    val t2 = TableType(TStruct("a" -> TInt32), FastIndexedSeq("a"), TStruct("g3" -> TInt32, "g4" -> TFloat64))
    val tab1 = TableLiteral(TableValue(ctx, t1, BroadcastRow(ctx, Row(1, 1.1), t1.globalType), RVD.empty(t1.canonicalRVDType)))
    val tab2 = TableLiteral(TableValue(ctx, t2, BroadcastRow(ctx, Row(2, 2.2), t2.globalType), RVD.empty(t2.canonicalRVDType)))

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

    val ir = RelationalLet("x", NA(TInt32), RelationalRef("x", TInt32))
    assertEvalsTo(ir, null)
  }


  @Test def testRelationalLetTable() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32))
    val ir = TableAggregate(RelationalLetTable("x",
      Literal(t, FastIndexedSeq(Row(1))),
      TableParallelize(MakeStruct(FastSeq("rows" -> RelationalRef("x", t), "global" -> MakeStruct(FastSeq()))))),
      ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(), AggSignature(Count(), FastIndexedSeq(), FastIndexedSeq())))
    assertEvalsTo(ir, 1L)
  }

  @Test def testRelationalLetMatrixTable() {
    implicit val execStrats = ExecStrategy.interpretOnly

    val t = TArray(TStruct("x" -> TInt32))
    val m = CastTableToMatrix(
      TableMapGlobals(
        TableMapRows(
          TableRange(1, 1), InsertFields(Ref("row", TStruct("idx" -> TInt32)), FastSeq("entries" -> RelationalRef("x", t)))),
        MakeStruct(FastSeq("cols" -> MakeArray(FastSeq(MakeStruct(FastSeq("s" -> I32(0)))), TArray(TStruct("s" -> TInt32)))))),
      "entries",
      "cols",
      FastIndexedSeq())
    val ir = MatrixAggregate(RelationalLetMatrixTable("x",
      Literal(t, FastIndexedSeq(Row(1))),
      m),
      ApplyAggOp(FastIndexedSeq(), FastIndexedSeq(), AggSignature(Count(), FastIndexedSeq(), FastIndexedSeq())))
    assertEvalsTo(ir, 1L)
  }


  @DataProvider(name = "relationalFunctions")
  def relationalFunctionsData(): Array[Array[Any]] = Array(
    Array(TableFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(VEP(fs, "src/test/resources/dummy_vep_config.json", false, 1)),
    Array(WrappedMatrixToTableFunction(LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b")), "foo", "bar", FastIndexedSeq("ck"))),
    Array(LinearRegressionRowsSingle(Array("foo"), "bar", Array("baz"), 1, Array("a", "b"))),
    Array(LinearRegressionRowsChained(FastIndexedSeq(FastIndexedSeq("foo")), "bar", Array("baz"), 1, Array("a", "b"))),
    Array(LogisticRegression("firth", Array("a", "b"), "c", Array("d", "e"), Array("f", "g"))),
    Array(PoissonRegression("firth", "a", "c", Array("d", "e"), Array("f", "g"))),
    Array(Skat("a", "b", "c", "d", Array("e", "f"), false, 1, 0.1, 100)),
    Array(LocalLDPrune("x", 0.95, 123, 456)),
    Array(PCA("x", 1, false)),
    Array(PCRelate(0.00, 4096, Some(0.1), PCRelate.PhiK2K0K1)),
    Array(MatrixFilterPartitions(Array(1, 2, 3), keep = true)),
    Array(ForceCountTable()),
    Array(ForceCountMatrixTable()),
    Array(NPartitionsTable()),
    Array(NPartitionsMatrixTable()),
    Array(WrappedMatrixToValueFunction(NPartitionsMatrixTable(), "foo", "bar", FastIndexedSeq("a", "c"))),
    Array(MatrixWriteBlockMatrix("a", false, "b", 1)),
    Array(MatrixExportEntriesByCol(1, "asd", false, true, false)),
    Array(GetElement(FastSeq(1, 2)))
  )

  @Test def relationalFunctionsRun(): Unit = {
    relationalFunctionsData()
  }

  @Test(dataProvider = "relationalFunctions")
  def testRelationalFunctionsSerialize(x: Any): Unit = {
    implicit val formats = RelationalFunctions.formats

    x match {
      case x: MatrixToMatrixFunction => assert(RelationalFunctions.lookupMatrixToMatrix(ctx, Serialization.write(x)) == x)
      case x: MatrixToTableFunction => assert(RelationalFunctions.lookupMatrixToTable(ctx, Serialization.write(x)) == x)
      case x: MatrixToValueFunction => assert(RelationalFunctions.lookupMatrixToValue(ctx, Serialization.write(x)) == x)
      case x: TableToTableFunction => assert(RelationalFunctions.lookupTableToTable(ctx, JsonMethods.compact(x.toJValue)) == x)
      case x: TableToValueFunction => assert(RelationalFunctions.lookupTableToValue(ctx, Serialization.write(x)) == x)
      case x: BlockMatrixToTableFunction => assert(RelationalFunctions.lookupBlockMatrixToTable(ctx, Serialization.write(x)) == x)
      case x: BlockMatrixToValueFunction => assert(RelationalFunctions.lookupBlockMatrixToValue(ctx, Serialization.write(x)) == x)
    }
  }

  @Test def testFoldWithSetup() {
    val v = In(0, TInt32)
    val cond1 = If(v.ceq(I32(3)),
      MakeStream(FastIndexedSeq(I32(1), I32(2), I32(3)), TStream(TInt32)),
      MakeStream(FastIndexedSeq(I32(4), I32(5), I32(6)), TStream(TInt32)))
    assertEvalsTo(StreamFold(cond1, True(), "accum", "i", Ref("i", TInt32).ceq(v)), FastIndexedSeq(0 -> TInt32), false)
  }

  @Test def testNonCanonicalTypeParsing(): Unit = {
    val t = TTuple(FastIndexedSeq(TupleField(1, TInt64)))
    val lit = Literal(t, Row(1L))

    assert(IRParser.parseType(t.parsableString()) == t)
    assert(IRParser.parse_value_ir(ctx, Pretty(lit, elideLiterals = false)) == lit)
  }

  def regressionTestUnifyBug(): Unit = {
    // failed due to misuse of Type.unify
    val ir = IRParser.parse_value_ir(ctx,
      """
        |(ToArray (StreamMap __uid_3
        |    (ToStream (Literal Array[Interval[Locus(GRCh37)]] "[{\"start\": {\"contig\": \"20\", \"position\": 10277621}, \"end\": {\"contig\": \"20\", \"position\": 11898992}, \"includeStart\": true, \"includeEnd\": false}]"))
        |    (Apply Interval Interval[Struct{locus:Locus(GRCh37)}]
        |       (MakeStruct (locus  (Apply start Locus(GRCh37) (Ref __uid_3))))
        |       (MakeStruct (locus  (Apply end Locus(GRCh37) (Ref __uid_3)))) (True) (False))))
        |""".stripMargin)
    val (v, _) = backend.execute(ir, optimize = true)
    assert(
      ir.typ.ordering.equiv(
        FastIndexedSeq(
          Interval(
            Row(Locus("20", 10277621)), Row(Locus("20", 11898992)), includesStart = true, includesEnd = false)),
        v))
  }

  @Test def testSimpleTailLoop(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val triangleSum: IR = TailLoop("f",
      FastIndexedSeq("x" -> In(0, TInt32), "accum" -> In(1, TInt32)),
      If(Ref("x", TInt32) <= I32(0),
        Ref("accum", TInt32),
        Recur("f",
          FastIndexedSeq(
            Ref("x", TInt32) - I32(1),
            Ref("accum", TInt32) + Ref("x", TInt32)),
          TInt32)))

    assertEvalsTo(triangleSum, FastIndexedSeq(5 -> TInt32, 0 -> TInt32), 15)
    assertEvalsTo(triangleSum, FastIndexedSeq(5 -> TInt32, (null, TInt32)), null)
    assertEvalsTo(triangleSum, FastIndexedSeq((null, TInt32),  0 -> TInt32), null)
  }

  @Test def testNestedTailLoop(): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val triangleSum: IR = TailLoop("f1",
      FastIndexedSeq("x" -> In(0, TInt32), "accum" -> I32(0)),
      If(Ref("x", TInt32) <= I32(0),
        TailLoop("f2",
          FastIndexedSeq("x2" -> Ref("accum", TInt32), "accum2" -> I32(0)),
          If(Ref("x2", TInt32) <= I32(0),
            Ref("accum2", TInt32),
            Recur("f2",
              FastIndexedSeq(
                Ref("x2", TInt32) - I32(5),
                Ref("accum2", TInt32) + Ref("x2", TInt32)),
              TInt32))),
        Recur("f1",
          FastIndexedSeq(
            Ref("x", TInt32) - I32(1),
            Ref("accum", TInt32) + Ref("x", TInt32)),
          TInt32)))

    assertEvalsTo(triangleSum, FastIndexedSeq(5 -> TInt32), 15 + 10 + 5)
  }

  @Test def testHasIRSharing(): Unit = {
    val r = Ref("x", TInt32)
    val ir1 = MakeTuple.ordered(FastSeq(I64(1), r, r, I32(1)))
    assert(HasIRSharing(ir1))
    assert(!HasIRSharing(ir1.deepCopy()))
  }

  @Test def freeVariablesAggScanBindingEnv(): Unit = {
    def testFreeVarsHelper(ir: IR): Unit = {
      val irFreeVarsTrue = FreeVariables.apply(ir, true, true)
      assert(irFreeVarsTrue.agg.isDefined && irFreeVarsTrue.scan.isDefined)

      val irFreeVarsFalse = FreeVariables.apply(ir, false, false)
      assert(irFreeVarsFalse.agg.isEmpty && irFreeVarsFalse.scan.isEmpty)
    }

    val liftIR = LiftMeOut(Ref("x", TInt32))
    testFreeVarsHelper(liftIR)

    val sumSig = AggSignature(Sum(), Seq(), Seq(TInt64))
    val streamAggIR =  StreamAgg(
      StreamMap(StreamRange(I32(0), I32(4), I32(1)), "x", Cast(Ref("x", TInt32), TInt64)),
      "x",
      ApplyAggOp(FastIndexedSeq.empty, FastIndexedSeq(Ref("x", TInt64)), sumSig))
    testFreeVarsHelper(streamAggIR)

    val streamScanIR = StreamAggScan(Ref("st", TStream(TInt32)), "x", ApplyScanOp(FastIndexedSeq.empty, FastIndexedSeq(Cast(Ref("x", TInt32), TInt64)), sumSig))
    testFreeVarsHelper(streamScanIR)
  }

  @DataProvider(name = "nonNullTypesAndValues")
  def nonNullTypesAndValues(): Array[Array[Any]] = Array(
    Array(PInt32(), 1),
    Array(PInt64(), 5L),
    Array(PFloat32(), 5.5f),
    Array(PFloat64(), 1.2),
    Array(PCanonicalString(), "foo"),
    Array(PCanonicalArray(PInt32()), FastIndexedSeq(5, 7, null, 3)),
    Array(PCanonicalTuple(false, PInt32(), PCanonicalString(), PCanonicalStruct()), Row(3, "bar", Row()))
  )

  @Test(dataProvider = "nonNullTypesAndValues")
  def testReadWriteValues(pt: PType, value: Any): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val node = In(0, pt)
    val spec = TypedCodecSpec(pt, BufferSpec.defaultUncompressed)
    val prefix = ctx.createTmpPath("test-read-write-values")
    val filename = WriteValue(node, Str(prefix), spec)
    for (v <- Array(value, null)) {
      assertEvalsTo(ReadValue(filename, spec, pt.virtualType), FastIndexedSeq(v -> pt.virtualType), v)
    }
  }

  @Test(dataProvider="nonNullTypesAndValues")
  def testReadWriteValueDistributed(pt: PType, value: Any): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val node = In(0, pt)
    val spec = TypedCodecSpec(pt, BufferSpec.defaultUncompressed)
    val prefix = ctx.createTmpPath("test-read-write-value-dist")
    val readArray = Let("files",
      CollectDistributedArray(StreamMap(StreamRange(0, 10, 1), "x", node), MakeStruct(FastSeq()),
        "ctx", "globals",
        WriteValue(Ref("ctx", node.typ), Str(prefix), spec)),
      StreamMap(ToStream(Ref("files", TArray(TString))), "filename",
        ReadValue(Ref("filename", TString), spec, pt.virtualType)))
    for (v <- Array(value, null)) {
      assertEvalsTo(ToArray(readArray), FastIndexedSeq(v -> pt.virtualType), Array.fill(10)(v).toFastIndexedSeq)
    }
  }

  @Test def testUUID4() {
    val single = UUID4()
    val hex = "[0-9a-f]"
    val format = s"$hex{8}-$hex{4}-$hex{4}-$hex{4}-$hex{12}"
    // 12345678-1234-5678-1234-567812345678
    assertEvalsTo(
      bindIR(single){ s =>
        invoke("regexMatch", TBoolean, Str(format), s) &&
          invoke("length", TInt32, s).ceq(I32(36))
      }, true)

    val stream = mapIR(rangeIR(5)) { _ => single }

    def selfZip(s: IR, n: Int) = StreamZip(Array.fill(n)(s), Array.tabulate(n)(i => s"$i"),
      MakeArray(Array.tabulate(n)(i => Ref(s"$i", TString)), TArray(TString)),
      ArrayZipBehavior.AssumeSameLength)

    def assertNumDistinct(s: IR, expected: Int) =
      assertEvalsTo(ArrayLen(CastToArray(ToSet(s))), expected)

    assertNumDistinct(stream, 5)
    assertNumDistinct(flatten(selfZip(stream, 2)), 10)
    assertNumDistinct(bindIR(ToArray(stream))(a => selfZip(ToStream(a), 2)), 5)
  }

  @Test def testZipDoesntPruneLengthInfo(): Unit = {
    for (behavior <- Array(ArrayZipBehavior.AssumeSameLength,
      ArrayZipBehavior.AssertSameLength,
      ArrayZipBehavior.TakeMinLength,
      ArrayZipBehavior.ExtendNA)) {
      val zip = StreamZip(
        FastIndexedSeq(StreamRange(0, 10, 1), StreamRange(0, 10, 1)),
        FastIndexedSeq("x", "y"),
        makestruct("x" -> Str("foo"), "y" -> Str("bar")),
        behavior)

      assertEvalsTo(ToArray(zip), Array.fill(10)(Row("foo", "bar")).toFastIndexedSeq)
    }
  }
}

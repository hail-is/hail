package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.expr._
import is.hail.types._
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.check.{Gen, Prop}
import is.hail.types.virtual._
import org.testng.annotations.Test
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.Call2
import is.hail.utils._
import is.hail.expr.ir.DeprecatedIRBuilder._
import is.hail.expr.ir.lowering.{DArrayLowering, LowerTableIR}
import org.apache.spark.sql.Row

class AggregatorsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.compileOnly

  def runAggregator(op: AggOp, aggType: TStruct, agg: IndexedSeq[Row], expected: Any, initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR]) {

    val aggSig = AggSignature(op, initOpArgs.map(_.typ), seqOpArgs.map(_.typ))
    assertEvalsTo(
      ApplyAggOp(initOpArgs, seqOpArgs, aggSig),
      (agg, aggType),
      expected)
  }

  def runAggregator(op: AggOp, t: Type, a: IndexedSeq[Any], expected: Any,
    initOpArgs: IndexedSeq[IR] = FastIndexedSeq()) {
    runAggregator(op,
      TStruct("x" -> t),
      a.map(i => Row(i)),
      expected,
      initOpArgs,
      seqOpArgs = FastIndexedSeq(Ref("x", t)))
  }

  @Test def sumFloat64() {
    runAggregator(Sum(), TFloat64, (0 to 100).map(_.toDouble), 5050.0)
    runAggregator(Sum(), TFloat64, FastIndexedSeq(), 0.0)
    runAggregator(Sum(), TFloat64, FastIndexedSeq(42.0), 42.0)
    runAggregator(Sum(), TFloat64, FastIndexedSeq(null, 42.0, null), 42.0)
    runAggregator(Sum(), TFloat64, FastIndexedSeq(null, null, null), 0.0)
  }

  @Test def sumInt64() {
    runAggregator(Sum(), TInt64, FastIndexedSeq(-1L, 2L, 3L), 4L)
  }

  @Test def collectBoolean() {
    runAggregator(Collect(), TBoolean, FastIndexedSeq(true, false, null, true, false), FastIndexedSeq(true, false, null, true, false))
  }

  @Test def collectInt() {
    runAggregator(Collect(), TInt32, FastIndexedSeq(10, null, 5), FastIndexedSeq(10, null, 5))
  }

  @Test def collectLong() {
    runAggregator(Collect(), TInt64, FastIndexedSeq(10L, null, 5L), FastIndexedSeq(10L, null, 5L))
  }

  @Test def collectFloat() {
    runAggregator(Collect(), TFloat32, FastIndexedSeq(10f, null, 5f), FastIndexedSeq(10f, null, 5f))
  }

  @Test def collectDouble() {
    runAggregator(Collect(), TFloat64, FastIndexedSeq(10d, null, 5d), FastIndexedSeq(10d, null, 5d))
  }

  @Test def collectString() {
    runAggregator(Collect(), TString, FastIndexedSeq("hello", null, "foo"), FastIndexedSeq("hello", null, "foo"))
  }

  @Test def collectArray() {
    runAggregator(Collect(),
      TArray(TInt32), FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq()), FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq()))
  }

  @Test def collectStruct() {
    runAggregator(Collect(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null))
  }

  @Test def count() {
    runAggregator(Count(),
      TStruct("x" -> TString),
      FastIndexedSeq(Row("hello"), Row("foo"), Row("a"), Row(null), Row("b"), Row(null), Row("c")),
      7L,
      initOpArgs = FastIndexedSeq(),
      seqOpArgs = FastIndexedSeq())
  }

  @Test def collectAsSetBoolean() {
    runAggregator(CollectAsSet(), TBoolean, FastIndexedSeq(true, false, null, true, false), Set(true, false, null))
    runAggregator(CollectAsSet(), TBoolean, FastIndexedSeq(true, null, true), Set(true, null))
  }

  @Test def collectAsSetNumeric() {
    runAggregator(CollectAsSet(), TInt32, FastIndexedSeq(10, null, 5, 5, null), Set(10, null, 5))
    runAggregator(CollectAsSet(), TInt64, FastIndexedSeq(10L, null, 5L, 5L, null), Set(10L, null, 5L))
    runAggregator(CollectAsSet(), TFloat32, FastIndexedSeq(10f, null, 5f, 5f, null), Set(10f, null, 5f))
    runAggregator(CollectAsSet(), TFloat64, FastIndexedSeq(10d, null, 5d, 5d, null), Set(10d, null, 5d))
  }

  @Test def collectAsSetString() {
    runAggregator(CollectAsSet(), TString, FastIndexedSeq("hello", null, "foo", null, "foo"), Set("hello", null, "foo"))
  }

  @Test def collectAsSetArray() {
    val inputCollection = FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq(), null, FastIndexedSeq(1, 2, 3))
    val expected = Set(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq())
    runAggregator(CollectAsSet(), TArray(TInt32), inputCollection, expected)
  }

  @Test def collectAsSetStruct(): Unit = {
    runAggregator(CollectAsSet(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null, Row(5, true)),
      Set(Row(5, true), Row(3, false), null, Row(0, false)))
  }

  @Test def callStats() {
    runAggregator(CallStats(), TCall,
      FastIndexedSeq(Call2(0, 0), Call2(0, 1), null, Call2(0, 2)),
      Row(FastIndexedSeq(4, 1, 1), FastIndexedSeq(4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), 6, FastIndexedSeq(1, 0, 0)),
      initOpArgs = FastIndexedSeq(I32(3)))
  }

  // FIXME Max Boolean not supported by old-style MaxAggregator

  @Test def maxInt32() {
    runAggregator(Max(), TInt32, FastIndexedSeq(), null)
    runAggregator(Max(), TInt32, FastIndexedSeq(null), null)
    runAggregator(Max(), TInt32, FastIndexedSeq(-2, null, 7), 7)
  }

  @Test def maxInt64() {
    runAggregator(Max(), TInt64, FastIndexedSeq(-2L, null, 7L), 7L)
  }

  @Test def maxFloat32() {
    runAggregator(Max(), TFloat32, FastIndexedSeq(-2.0f, null, 7.2f), 7.2f)
  }

  @Test def maxFloat64() {
    runAggregator(Max(), TFloat64, FastIndexedSeq(-2.0, null, 7.2), 7.2)
  }

  @Test def takeInt32() {
    runAggregator(Take(), TInt32, FastIndexedSeq(2, null, 7), FastIndexedSeq(2, null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeInt64() {
    runAggregator(Take(), TInt64, FastIndexedSeq(2L, null, 7L), FastIndexedSeq(2L, null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat32() {
    runAggregator(Take(), TFloat32, FastIndexedSeq(2.0f, null, 7.2f), FastIndexedSeq(2.0f, null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat64() {
    runAggregator(Take(), TFloat64, FastIndexedSeq(2.0, null, 7.2), FastIndexedSeq(2.0, null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeCall() {
    runAggregator(Take(), TCall, FastIndexedSeq(Call2(0, 0), null, Call2(1, 0)), FastIndexedSeq(Call2(0, 0), null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeString() {
    runAggregator(Take(), TString, FastIndexedSeq("a", null, "b"), FastIndexedSeq("a", null),
      initOpArgs = FastIndexedSeq(I32(2)))
  }

  @Test
  def sumMultivar() {
    val aggSig = AggSignature(Sum(), FastSeq(), FastSeq(TFloat64))
    assertEvalsTo(ApplyAggOp(
      FastSeq(),
      FastSeq(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64), Ref("b", TFloat64))),
      aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64, "b" -> TFloat64)),
      110.0)
  }

  private[this] def assertArraySumEvalsTo[T](
    eltType: Type,
    a: IndexedSeq[Seq[T]],
    expected: Seq[T]
  ): Unit = {
    val aggSig = AggSignature(Sum(), FastSeq(), FastSeq(eltType))

    val aggregable = a.map(Row(_))
    val structType = TStruct("foo" -> TArray(eltType))

    assertEvalsTo(
        AggArrayPerElement(Ref("foo", TArray(eltType)), "elt", "_",
          ApplyAggOp(FastSeq(), FastSeq(Ref("elt", eltType)), aggSig), None, isScan = false),
      (aggregable, structType),
      expected)
  }

  @Test
  def arraySumFloat64OnEmpty(): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastIndexedSeq(),
      null
    )

  @Test
  def arraySumFloat64OnSingletonMissing(): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastIndexedSeq(null),
      null
    )

  @Test
  def arraySumFloat64OnAllMissing(): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastIndexedSeq(null, null, null),
      null
    )

  @Test
  def arraySumInt64OnEmpty(): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastIndexedSeq(),
      null
    )

  @Test
  def arraySumInt64OnSingletonMissing(): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastIndexedSeq(null),
      null
    )

  @Test
  def arraySumInt64OnAllMissing(): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastIndexedSeq(null, null, null),
      null
    )

  @Test
  def arraySumFloat64OnSmallArray(): Unit =
    assertArraySumEvalsTo(
      TFloat64,
      FastIndexedSeq(
        FastSeq(1.0, 2.0),
        FastSeq(10.0, 20.0),
        null),
      FastSeq(11.0, 22.0)
    )

  @Test
  def arraySumInt64OnSmallArray(): Unit =
    assertArraySumEvalsTo(
      TInt64,
      FastIndexedSeq(
        FastSeq(1L, 2L),
        FastSeq(10L, 20L),
        null),
      FastSeq(11L, 22L)
    )

  @Test
  def arraySumInt64FirstElementMissing(): Unit =
    assertArraySumEvalsTo(
      TInt64,
      FastIndexedSeq(
        null,
        FastSeq(1L, 33L),
        FastSeq(42L, 3L)),
      FastSeq(43L, 36L)
    )

  private[this] def assertTakeByEvalsTo(aggType: Type, keyType: Type, n: Int, a: IndexedSeq[Row], expected: IndexedSeq[Any]) {
    runAggregator(TakeBy(), TStruct("x" -> aggType, "y" -> keyType),
      a,
      expected,
      initOpArgs = FastIndexedSeq(I32(n)),
      seqOpArgs = FastIndexedSeq(Ref("x", aggType), Ref("y", keyType)))
  }

  @Test def takeByNGreater() {
    assertTakeByEvalsTo(TInt32, TInt32, 5,
      FastIndexedSeq(Row(3, 4)),
      FastIndexedSeq(3))
  }

  @Test def takeByBooleanBoolean() {
    assertTakeByEvalsTo(TBoolean, TBoolean, 3,
      FastIndexedSeq(Row(false, true), Row(null, null), Row(true, false)),
      FastIndexedSeq(true, false, null))
  }

  @Test def takeByBooleanInt() {
    assertTakeByEvalsTo(TBoolean, TInt32, 3,
      FastIndexedSeq(Row(false, 0), Row(null, null), Row(true, 1), Row(false, 3), Row(true, null), Row(null, 2)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanLong() {
    assertTakeByEvalsTo(TBoolean, TInt64, 3,
      FastIndexedSeq(Row(false, 0L), Row(null, null), Row(true, 1L), Row(false, 3L), Row(true, null), Row(null, 2L)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanFloat() {
    assertTakeByEvalsTo(TBoolean, TFloat32, 3,
      FastIndexedSeq(Row(false, 0F), Row(null, null), Row(true, 1F), Row(false, 3F), Row(true, null), Row(null, 2F)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanDouble() {
    assertTakeByEvalsTo(TBoolean, TFloat64, 3,
      FastIndexedSeq(Row(false, 0D), Row(null, null), Row(true, 1D), Row(false, 3D), Row(true, null), Row(null, 2D)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanAnnotation() {
    assertTakeByEvalsTo(TBoolean, TString, 3,
      FastIndexedSeq(Row(false, "a"), Row(null, null), Row(true, "b"), Row(false, "d"), Row(true, null), Row(null, "c")),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByIntBoolean() {
    assertTakeByEvalsTo(TInt32, TBoolean, 2,
      FastIndexedSeq(Row(3, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3))
  }

  @Test def takeByIntInt() {
    assertTakeByEvalsTo(TInt32, TInt32, 3,
      FastIndexedSeq(Row(3, 4), Row(null, null), Row(null, 2), Row(11, 0), Row(45, 1), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntLong() {
    assertTakeByEvalsTo(TInt32, TInt64, 3,
      FastIndexedSeq(Row(3, 4L), Row(null, null), Row(null, 2L), Row(11, 0L), Row(45, 1L), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntFloat() {
    assertTakeByEvalsTo(TInt32, TFloat32, 3,
      FastIndexedSeq(Row(3, 4F), Row(null, null), Row(null, 2F), Row(11, 0F), Row(45, 1F), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntDouble() {
    assertTakeByEvalsTo(TInt32, TFloat64, 3,
      FastIndexedSeq(Row(3, 4D), Row(null, null), Row(null, 2D), Row(11, 0D), Row(45, 1D), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntAnnotation() {
    assertTakeByEvalsTo(TInt32, TString, 3,
      FastIndexedSeq(Row(3, "d"), Row(null, null), Row(null, "c"), Row(11, "a"), Row(45, "b"), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByLongBoolean() {
    assertTakeByEvalsTo(TInt64, TBoolean, 2,
      FastIndexedSeq(Row(3L, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3L))
  }

  @Test def takeByLongInt() {
    assertTakeByEvalsTo(TInt64, TInt32, 3,
      FastIndexedSeq(Row(3L, 4), Row(null, null), Row(null, 2), Row(11L, 0), Row(45L, 1), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongLong() {
    assertTakeByEvalsTo(TInt64, TInt64, 3,
      FastIndexedSeq(Row(3L, 4L), Row(null, null), Row(null, 2L), Row(11L, 0L), Row(45L, 1L), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongFloat() {
    assertTakeByEvalsTo(TInt64, TFloat32, 3,
      FastIndexedSeq(Row(3L, 4F), Row(null, null), Row(null, 2F), Row(11L, 0F), Row(45L, 1F), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongDouble() {
    assertTakeByEvalsTo(TInt64, TFloat64, 3,
      FastIndexedSeq(Row(3L, 4D), Row(null, null), Row(null, 2D), Row(11L, 0D), Row(45L, 1D), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongAnnotation() {
    assertTakeByEvalsTo(TInt64, TString, 3,
      FastIndexedSeq(Row(3L, "d"), Row(null, null), Row(null, "c"), Row(11L, "a"), Row(45L, "b"), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByFloatBoolean() {
    assertTakeByEvalsTo(TFloat32, TBoolean, 2,
      FastIndexedSeq(Row(3F, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3F))
  }

  @Test def takeByFloatInt() {
    assertTakeByEvalsTo(TFloat32, TInt32, 3,
      FastIndexedSeq(Row(3F, 4), Row(null, null), Row(null, 2), Row(11F, 0), Row(45F, 1), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatLong() {
    assertTakeByEvalsTo(TFloat32, TInt64, 3,
      FastIndexedSeq(Row(3F, 4L), Row(null, null), Row(null, 2L), Row(11F, 0L), Row(45F, 1L), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatFloat() {
    assertTakeByEvalsTo(TFloat32, TFloat32, 3,
      FastIndexedSeq(Row(3F, 4F), Row(null, null), Row(null, 2F), Row(11F, 0F), Row(45F, 1F), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatDouble() {
    assertTakeByEvalsTo(TFloat32, TFloat64, 3,
      FastIndexedSeq(Row(3F, 4D), Row(null, null), Row(null, 2D), Row(11F, 0D), Row(45F, 1D), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatAnnotation() {
    assertTakeByEvalsTo(TFloat32, TString, 3,
      FastIndexedSeq(Row(3F, "d"), Row(null, null), Row(null, "c"), Row(11F, "a"), Row(45F, "b"), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByDoubleBoolean() {
    assertTakeByEvalsTo(TFloat64, TBoolean, 2,
      FastIndexedSeq(Row(3D, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3D))
  }

  @Test def takeByDoubleInt() {
    assertTakeByEvalsTo(TFloat64, TInt32, 3,
      FastIndexedSeq(Row(3D, 4), Row(null, null), Row(null, 2), Row(11D, 0), Row(45D, 1), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleLong() {
    assertTakeByEvalsTo(TFloat64, TInt64, 3,
      FastIndexedSeq(Row(3D, 4L), Row(null, null), Row(null, 2L), Row(11D, 0L), Row(45D, 1L), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleFloat() {
    assertTakeByEvalsTo(TFloat64, TFloat32, 3,
      FastIndexedSeq(Row(3D, 4F), Row(null, null), Row(null, 2F), Row(11D, 0F), Row(45D, 1F), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleDouble() {
    assertTakeByEvalsTo(TFloat64, TFloat64, 3,
      FastIndexedSeq(Row(3D, 4D), Row(null, null), Row(null, 2D), Row(11D, 0D), Row(45D, 1D), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleAnnotation() {
    assertTakeByEvalsTo(TFloat64, TString, 3,
      FastIndexedSeq(Row(3D, "d"), Row(null, null), Row(null, "c"), Row(11D, "a"), Row(45D, "b"), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByAnnotationBoolean() {
    assertTakeByEvalsTo(TString, TBoolean, 2,
      FastIndexedSeq(Row("hello", true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, "hello"))
  }

  @Test def takeByAnnotationInt() {
    assertTakeByEvalsTo(TString, TInt32, 3,
      FastIndexedSeq(Row("a", 4), Row(null, null), Row(null, 2), Row("b", 0), Row("c", 1), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationLong() {
    assertTakeByEvalsTo(TString, TInt64, 3,
      FastIndexedSeq(Row("a", 4L), Row(null, null), Row(null, 2L), Row("b", 0L), Row("c", 1L), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationFloat() {
    assertTakeByEvalsTo(TString, TFloat32, 3,
      FastIndexedSeq(Row("a", 4F), Row(null, null), Row(null, 2F), Row("b", 0F), Row("c", 1F), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationDouble() {
    assertTakeByEvalsTo(TString, TFloat64, 3,
      FastIndexedSeq(Row("a", 4D), Row(null, null), Row(null, 2D), Row("b", 0D), Row("c", 1D), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationAnnotation() {
    assertTakeByEvalsTo(TString, TString, 3,
      FastIndexedSeq(Row("a", "d"), Row(null, null), Row(null, "c"), Row("b", "a"), Row("c", "b"), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByCallLong() {
    assertTakeByEvalsTo(TCall, TInt64, 3,
      FastIndexedSeq(Row(Call2(0, 0), 4L), Row(null, null), Row(null, 2L), Row(Call2(0, 1), 0L), Row(Call2(1, 1), 1L), Row(Call2(0, 2), null)),
      FastIndexedSeq(Call2(0, 1), Call2(1, 1), null))
  }

  def runKeyedAggregator(
    op: AggOp,
    key: IR,
    aggType: TStruct,
    agg: IndexedSeq[Row],
    expected: Any,
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR]) {
    assertEvalsTo(
      AggGroupBy(key,
        ApplyAggOp(
          initOpArgs,
          seqOpArgs,
          AggSignature(op, initOpArgs.map(_.typ), seqOpArgs.map(_.typ))),
        false),
      (agg, aggType),
      expected)
  }

  @Test
  def keyedCount() {
    runKeyedAggregator(Count(),
      Ref("k", TInt32),
      TStruct("k" -> TInt32),
      FastIndexedSeq(Row(1), Row(2), Row(3), Row(1), Row(1), Row(null), Row(null)),
      Map(1 -> 3L, 2 -> 1L, 3 -> 1L, (null, 2L)),
      initOpArgs = FastIndexedSeq(),
      seqOpArgs = FastIndexedSeq())

    runKeyedAggregator(Count(),
      Ref("k", TBoolean),
      TStruct("k" -> TBoolean),
      FastIndexedSeq(Row(true), Row(true), Row(true), Row(false), Row(false), Row(null), Row(null)),
      Map(true -> 3L, false -> 2L, (null, 2L)),
      initOpArgs = FastIndexedSeq(),
      seqOpArgs = FastIndexedSeq())

    // test struct as key
    runKeyedAggregator(Count(),
      Ref("k", TStruct("a" -> TBoolean)),
      TStruct("k" -> TStruct("a" -> TBoolean)),
      FastIndexedSeq(Row(Row(true)), Row(Row(true)), Row(Row(true)), Row(Row(false)), Row(Row(false)), Row(Row(null)), Row(Row(null))),
      Map(Row(true) -> 3L, Row(false) -> 2L, (Row(null), 2L)),
      initOpArgs = FastIndexedSeq(),
      seqOpArgs = FastIndexedSeq())
  }

  @Test
  def keyedCollect() {
    runKeyedAggregator(
      Collect(),
      Ref("k", TBoolean),
      TStruct("k" -> TBoolean, "v" -> TInt32),
      FastIndexedSeq(Row(true, 5), Row(true, 3), Row(true, null), Row(false, 0), Row(false, null), Row(null, null), Row(null, 2)),
      Map(true -> FastIndexedSeq(5, 3, null), false -> FastIndexedSeq(0, null), (null, FastIndexedSeq(null, 2))),
      FastIndexedSeq(),
      FastIndexedSeq(Ref("v", TInt32)))
  }

  @Test
  def keyedCallStats() {
    runKeyedAggregator(
      CallStats(),
      Ref("k", TBoolean),
      TStruct("k" -> TBoolean, "v" ->TCall),
      FastIndexedSeq(Row(true, null), Row(true, Call2(0, 1)), Row(true, Call2(0, 1)),
        Row(false, null), Row(false, Call2(0, 0)), Row(false, Call2(1, 1))),
      Map(true -> Row(FastIndexedSeq(2, 2), FastIndexedSeq(0.5, 0.5), 4, FastIndexedSeq(0, 0)),
        false -> Row(FastIndexedSeq(2, 2), FastIndexedSeq(0.5, 0.5), 4, FastIndexedSeq(1, 1))),
      FastIndexedSeq(I32(2)),
      FastIndexedSeq(Ref("v", TCall)))
  }

  @Test
  def keyedTakeBy() {
    runKeyedAggregator(TakeBy(),
      Ref("k", TString),
      TStruct("k" -> TString, "x" -> TFloat64, "y" -> TInt32),
      FastIndexedSeq(Row("case", 0.2, 5), Row("control", 0.4, 0), Row(null, 1.0, 3), Row("control", 0.0, 2), Row("case", 0.3, 6), Row("control", 0.5, 1)),
      Map("case" -> FastIndexedSeq(0.2, 0.3),
        "control" -> FastIndexedSeq(0.4, 0.5),
        (null, FastIndexedSeq(1.0))),
      FastIndexedSeq(I32(2)),
      FastIndexedSeq(Ref("x", TFloat64), Ref("y", TInt32)))
  }

  @Test
  def keyedKeyedCollect() {
    val agg = FastIndexedSeq(Row("EUR", true, 1), Row("EUR", false, 2), Row("AFR", true, 3), Row("AFR", null, 4))
    val aggType = TStruct("k1" -> TString, "k2" -> TBoolean, "x" -> TInt32)
    val expected = Map("EUR" -> Map(true -> FastIndexedSeq(1), false -> FastIndexedSeq(2)), "AFR" -> Map(true -> FastIndexedSeq(3), (null, FastIndexedSeq(4))))
    val aggSig = AggSignature(Collect(), FastIndexedSeq(), FastIndexedSeq(TInt32))
    assertEvalsTo(
      AggGroupBy(Ref("k1", TString),
        AggGroupBy(Ref("k2", TBoolean),
          ApplyAggOp(
            FastSeq(),
            FastSeq(Ref("x", TInt32)),
            aggSig),
          false),
        false),
      (agg, aggType),
      expected
    )
  }

  @Test
  def keyedKeyedCallStats() {
    val agg = FastIndexedSeq(
      Row("EUR", "CASE", null),
      Row("EUR", "CONTROL", Call2(0, 1)),
      Row("AFR", "CASE", Call2(1, 1)),
      Row("AFR", "CONTROL", null))
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "g" -> TCall)
    val expected = Map(
      "EUR" -> Map(
        "CONTROL" -> Row(FastIndexedSeq(1, 1), FastIndexedSeq(0.5, 0.5), 2, FastIndexedSeq(0, 0)),
        "CASE" -> Row(FastIndexedSeq(0, 0), null, 0, FastIndexedSeq(0, 0))),
      "AFR" -> Map(
        "CASE" -> Row(FastIndexedSeq(0, 2), FastIndexedSeq(0.0, 1.0), 2, FastIndexedSeq(0, 1)),
        "CONTROL" -> Row(FastIndexedSeq(0, 0), null, 0, FastIndexedSeq(0, 0))))
    val aggSig = AggSignature(CallStats(), FastIndexedSeq(TInt32), FastIndexedSeq(TCall))
    assertEvalsTo(
      AggGroupBy(Ref("k1", TString),
        AggGroupBy(Ref("k2", TString),
          ApplyAggOp(
            FastSeq(I32(2)),
            FastSeq(Ref("g", TCall)),
            aggSig), false), false),
      (agg, aggType),
      expected
    )
  }

  @Test
  def keyedKeyedTakeBy() {
    val agg = FastIndexedSeq(
      Row("case", "a", 0.2, 5), Row("control", "b", 0.4, 0),
      Row(null, "c", 1.0, 3), Row("control", "b", 0.0, 2),
      Row("case", "a", 0.3, 6), Row("control", "b", 0.5, 1))
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "x" -> TFloat64, "y" -> TInt32)
    val expected = Map(
      "case" -> Map("a" -> FastIndexedSeq(0.2, 0.3)),
      "control" -> Map("b" -> FastIndexedSeq(0.4, 0.5)),
      (null, Map("c" -> FastIndexedSeq(1.0))))
    val aggSig = AggSignature(TakeBy(), FastIndexedSeq(TInt32), FastIndexedSeq(TFloat64, TInt32))
    assertEvalsTo(
      AggGroupBy(Ref("k1", TString),
        AggGroupBy(Ref("k2", TString),
          ApplyAggOp(
            FastIndexedSeq(I32(2)),
            FastSeq(Ref("x", TFloat64), Ref("y", TInt32)),
            aggSig), false), false),
      (agg, aggType),
      expected
    )
  }

  @Test
  def keyedKeyedKeyedCollect() {
    val agg = FastIndexedSeq(Row("EUR", "CASE", true, 1), Row("EUR", "CONTROL", true, 2), Row("AFR", "CASE", false, 3), Row("AFR", "CONTROL", false, 4))
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "k3" -> TBoolean, "x" -> TInt32)
    val expected = Map("EUR" -> Map("CASE" -> Map(true -> FastIndexedSeq(1)), "CONTROL" -> Map(true -> FastIndexedSeq(2))), "AFR" -> Map("CASE" -> Map(false -> FastIndexedSeq(3)), "CONTROL" -> Map(false -> FastIndexedSeq(4))))
    val aggSig = AggSignature(Collect(), FastIndexedSeq(), FastIndexedSeq(TInt32))
    assertEvalsTo(
      AggGroupBy(Ref("k1", TString),
        AggGroupBy(Ref("k2", TString),
          AggGroupBy(Ref("k3", TBoolean),
            ApplyAggOp(
              FastSeq(),
              FastSeq(Ref("x", TInt32)),
              aggSig), false), false), false),
      (agg, aggType),
      expected
    )
  }

  @Test def downsampleWhenEmpty(): Unit = {
    runAggregator(Downsample(),
      TStruct("x" -> TFloat64, "y" -> TFloat64, "label" -> TArray(TString)),
      FastIndexedSeq(),
      FastIndexedSeq(),
      FastIndexedSeq(10),
      seqOpArgs = FastIndexedSeq(Ref("x", TFloat64), Ref("y", TFloat64), Ref("label", TArray(TString))))
  }

  @Test def testAggFilter(): Unit = {
    val aggSig = AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64))
    val aggType = TStruct("x" -> TBoolean, "y" -> TInt64)
    val agg = FastIndexedSeq(Row(true, -1L), Row(true, 1L), Row(false, 3L), Row(true, 5L))

    assertEvalsTo(
          AggFilter(Ref("x", TBoolean),
            ApplyAggOp(FastSeq(),
              FastSeq(Ref("y", TInt64)),
              aggSig), false),
      (agg, aggType),
      5L)
  }

  @Test def testAggExplode(): Unit = {
    val aggSig = AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64))
    val aggType = TStruct("x" -> TArray(TInt64))
    val agg = FastIndexedSeq(
      Row(FastIndexedSeq[Long](1, 4)),
      Row(FastIndexedSeq[Long]()),
      Row(FastIndexedSeq[Long](-1, 3)),
      Row(FastIndexedSeq[Long](4, 5, 6, -7)))

    assertEvalsTo(
      AggExplode(ToStream(Ref("x", TArray(TInt64))),
        "y",
        ApplyAggOp(FastSeq(),
          FastSeq(Ref("y", TInt64)),
          aggSig), false),
      (agg, aggType),
      15L)
  }

  @Test def testArrayElementsAggregator(): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int): IR = {
      hc
      val ht = TableRange(10, 3)
        .mapRows('row.insertFields('aRange -> irRange(0, m, 1)))

      TableAggregate(
        ht,
        AggArrayPerElement(GetField(Ref("row", ht.typ.rowType), "aRange"), "elt", "_'",
          ApplyAggOp(
            FastIndexedSeq(),
            FastIndexedSeq(Cast(Ref("elt", TInt32), TInt64)),
            AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64))),
          None,
          false
        )
      )
    }

    assertEvalsTo(getAgg(10, 10), IndexedSeq.range(0, 10).map(_ * 10L))
  }

  @Test def testArrayElementsAggregatorEmpty(): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int, knownLength: Option[IR]): IR = {
      hc
      val ht = TableRange(10, 3)
        .mapRows('row.insertFields('aRange -> irRange(0, m, 1)))
        .mapGlobals('global.insertFields('m -> m))
        .filter(false)

      TableAggregate(
        ht,
        AggArrayPerElement(GetField(Ref("row", ht.typ.rowType), "aRange"), "elt", "_'",
          ApplyAggOp(
            FastIndexedSeq(),
            FastIndexedSeq(Cast(Ref("elt", TInt32), TInt64)),
            AggSignature(Sum(), FastIndexedSeq(), FastIndexedSeq(TInt64))),
          knownLength,
          false
        )
      )
    }

    assertEvalsTo(getAgg(10, 10, None), null)
    assertEvalsTo(getAgg(10, 10, Some(1)), FastIndexedSeq(0L))
    assertEvalsTo(getAgg(10, 10, Some(GetField(Ref("global", TStruct("m" -> TInt32)), "m"))), Array.fill(10)(0L).toFastIndexedSeq)
  }

  @Test def testImputeTypeSimple(): Unit = {
    runAggregator(ImputeType(), TString, FastIndexedSeq(null), Row(false, false, true, true, true, true))
    runAggregator(ImputeType(), TString, FastIndexedSeq("1231", "1234.5", null), Row(true, false, false, false, false, true))
    runAggregator(ImputeType(), TString, FastIndexedSeq("1231", "123"), Row(true, true, false, true, true, true))
    runAggregator(ImputeType(), TString, FastIndexedSeq("true", "false"), Row(true, true, true, false, false, false))
  }

  @Test def testFoldAgg(): Unit = {
    val barRef = Ref("bar", TInt32)
    val bazRef = Ref("baz", TInt32)

    val myIR = StreamAgg(mapIR(rangeIR(100)){ idx => makestruct(("idx", idx), ("unused", idx + idx))}, "foo",
      AggFold(I32(0), Ref("bar", TInt32) + GetField(Ref("foo", TStruct("idx" -> TInt32, "unused" -> TInt32)), "idx"), barRef + bazRef, "bar", "baz", false)
    )
    assertEvalsTo(myIR, 4950)

    val myTableIR = TableAggregate(TableRange(100, 5),
      AggFold(I32(0), Ref("bar", TInt32) + GetField(Ref("row", TStruct("idx" -> TInt32)), "idx"), barRef + bazRef, "bar", "baz", false)
    )

    val analyses = LoweringAnalyses.apply(myTableIR, ctx)
    val myLoweredTableIR = LowerTableIR(myTableIR, DArrayLowering.All, ctx, analyses)

    assertEvalsTo(myLoweredTableIR, 4950)
  }

  @Test def testFoldScan(): Unit = {
    val barRef = Ref("bar", TInt32)
    val bazRef = Ref("baz", TInt32)

    val myIR = ToArray(StreamAggScan(mapIR(rangeIR(10)){ idx => makestruct(("idx", idx), ("unused", idx + idx))}, "foo",
      AggFold(I32(0), Ref("bar", TInt32) + GetField(Ref("foo", TStruct("idx" -> TInt32, "unused" -> TInt32)), "idx"), barRef + bazRef, "bar", "baz", true)
    ))
    assertEvalsTo(myIR, IndexedSeq(0, 0, 1, 3, 6, 10, 15, 21, 28, 36))
  }
}

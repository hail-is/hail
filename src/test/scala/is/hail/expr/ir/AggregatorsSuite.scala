package is.hail.expr.ir

import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.TestUtils._
import org.testng.annotations.Test
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.Call2
import org.apache.spark.sql.Row

class AggregatorsSuite {
  def runAggregator(op: AggOp, aggType: TStruct, agg: IndexedSeq[Row], expected: Any, constrArgs: IndexedSeq[IR],
    initOpArgs: Option[IndexedSeq[IR]], seqOpArgs: IndexedSeq[IR]) {

    val aggSig = AggSignature(op, constrArgs.map(_.typ), initOpArgs.map(_.map(_.typ)), seqOpArgs.map(_.typ))

    assertEvalsTo(ApplyAggOp(
      SeqOp(I32(0), seqOpArgs, aggSig),
      constrArgs, initOpArgs, aggSig),
      (agg, aggType),
      expected)
  }

  def runAggregator(op: AggOp, t: Type, a: IndexedSeq[Any], expected: Any,
    constrArgs: IndexedSeq[IR] = FastIndexedSeq(), initOpArgs: Option[IndexedSeq[IR]] = None) {
    runAggregator(op,
      TStruct("x" -> t),
      a.map(i => Row(i)),
      expected,
      constrArgs,
      initOpArgs,
      seqOpArgs = FastIndexedSeq(Ref("x", t)))
  }

  @Test def sumFloat64() {
    runAggregator(Sum(), TFloat64(), (0 to 100).map(_.toDouble), 5050.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(), 0.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(42.0), 42.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(null, 42.0, null), 42.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(null, null, null), 0.0)
  }

  @Test def sumInt64() {
    runAggregator(Sum(), TInt64(), FastIndexedSeq(-1L, 2L, 3L), 4L)

  }

  @Test def fraction() {
    runAggregator(Fraction(), TBoolean(), FastIndexedSeq(true, false, null, true, false), 2.0 / 5.0)
  }

  @Test def collectBoolean() {
    runAggregator(Collect(), TBoolean(), FastIndexedSeq(true, false, null, true, false), FastIndexedSeq(true, false, null, true, false))
  }

  @Test def collectInt() {
    runAggregator(Collect(), TInt32(), FastIndexedSeq(10, null, 5), FastIndexedSeq(10, null, 5))
  }

  @Test def collectLong() {
    runAggregator(Collect(), TInt64(), FastIndexedSeq(10L, null, 5L), FastIndexedSeq(10L, null, 5L))
  }

  @Test def collectFloat() {
    runAggregator(Collect(), TFloat32(), FastIndexedSeq(10f, null, 5f), FastIndexedSeq(10f, null, 5f))
  }

  @Test def collectDouble() {
    runAggregator(Collect(), TFloat64(), FastIndexedSeq(10d, null, 5d), FastIndexedSeq(10d, null, 5d))
  }

  @Test def collectString() {
    runAggregator(Collect(), TString(), FastIndexedSeq("hello", null, "foo"), FastIndexedSeq("hello", null, "foo"))
  }

  @Test def collectArray() {
    runAggregator(Collect(),
      TArray(TInt32()), FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq()), FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq()))
  }

  @Test def collectStruct() {
    runAggregator(Collect(),
      TStruct("a" -> TInt32(), "b" -> TBoolean()),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null))
  }

  @Test def count() {
    runAggregator(Count(),
      TStruct("x" -> TString()),
      FastIndexedSeq(Row("hello"), Row("foo"), Row("a"), Row(null), Row("b"), Row(null), Row("c")),
      7L,
      constrArgs = FastIndexedSeq(),
      initOpArgs = None,
      seqOpArgs = FastIndexedSeq())
  }

  @Test def counterString() {
    runAggregator(Counter(),
      TString(),
      FastIndexedSeq("rabbit", "rabbit", null, "cat", "dog", null),
      Map("rabbit" -> 2L, "cat" -> 1L, "dog" -> 1L, (null, 2L)))
  }

  @Test def counterArray() {
    runAggregator(Counter(),
      TArray(TInt32()),
      FastIndexedSeq(FastIndexedSeq(), FastIndexedSeq(1, 2, 3), null, FastIndexedSeq(), FastIndexedSeq(1), null),
      Map(FastIndexedSeq() -> 2L, FastIndexedSeq(1, 2, 3) -> 1L, FastIndexedSeq(1) -> 1L, (null, 2L)))
  }

  @Test def counterBoolean() {
    runAggregator(Counter(),
      TBoolean(),
      FastIndexedSeq(true, true, null, false, false, false, null),
      Map(true -> 2L, false -> 3L, (null, 2L)))

    runAggregator(Counter(),
      TBoolean(),
      FastIndexedSeq(true, true, false, false, false),
      Map(true -> 2L, false -> 3L))
  }

  @Test def counterInt() {
    runAggregator(Counter(),
      TInt32(),
      FastIndexedSeq(1, 3, null, 1, 4, 3, null),
      Map(1 -> 2L, 3 -> 2L, 4 -> 1L, (null, 2L)))
  }

  @Test def counterLong() {
    runAggregator(Counter(),
      TInt64(),
      FastIndexedSeq(1L, 3L, null, 1L, 4L, 3L, null),
      Map(1L -> 2L, 3L -> 2L, 4L -> 1L, (null, 2L)))
  }

  @Test def counterFloat() {
    runAggregator(Counter(),
      TFloat32(),
      FastIndexedSeq(1f, 3f, null, 1f, 4f, 3f, null),
      Map(1f -> 2L, 3f -> 2L, 4f -> 1L, (null, 2L)))
  }

  @Test def counterDouble() {
    runAggregator(Counter(),
      TFloat64(),
      FastIndexedSeq(1D, 3D, null, 1D, 4D, 3D, null),
      Map(1D -> 2L, 3D -> 2L, 4D -> 1L, (null, 2L)))
  }

  @Test def counterCall() {
    runAggregator(Counter(),
      TCall(),
      FastIndexedSeq(Call2(0, 0), Call2(0, 0), null, Call2(0, 1), Call2(1, 1), Call2(0, 0), null),
      Map(Call2(0, 0) -> 3L, Call2(0, 1) -> 1L, Call2(1, 1) -> 1L, (null, 2L)))
  }

  @Test def collectAsSetBoolean() {
    runAggregator(CollectAsSet(), TBoolean(), FastIndexedSeq(true, false, null, true, false), Set(true, false, null))
    runAggregator(CollectAsSet(), TBoolean(), FastIndexedSeq(true, null, true), Set(true, null))
  }

  @Test def collectAsSetNumeric() {
    runAggregator(CollectAsSet(), TInt32(), FastIndexedSeq(10, null, 5, 5, null), Set(10, null, 5))
    runAggregator(CollectAsSet(), TInt64(), FastIndexedSeq(10L, null, 5L, 5L, null), Set(10L, null, 5L))
    runAggregator(CollectAsSet(), TFloat32(), FastIndexedSeq(10f, null, 5f, 5f, null), Set(10f, null, 5f))
    runAggregator(CollectAsSet(), TFloat64(), FastIndexedSeq(10d, null, 5d, 5d, null), Set(10d, null, 5d))
  }

  @Test def collectAsSetString() {
    runAggregator(CollectAsSet(), TString(), FastIndexedSeq("hello", null, "foo", null, "foo"), Set("hello", null, "foo"))
  }

  @Test def collectAsSetArray() {
    val inputCollection = FastIndexedSeq(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq(), null, FastIndexedSeq(1, 2, 3))
    val expected = Set(FastIndexedSeq(1, 2, 3), null, FastIndexedSeq())
    runAggregator(CollectAsSet(), TArray(TInt32()), inputCollection, expected)
  }

  @Test def collectAsSetStruct(): Unit = {
    runAggregator(CollectAsSet(),
      TStruct("a" -> TInt32(), "b" -> TBoolean()),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null, Row(5, true)),
      Set(Row(5, true), Row(3, false), null, Row(0, false)))
  }

  @Test def callStats() {
    runAggregator(CallStats(), TCall(),
      FastIndexedSeq(Call2(0, 0), Call2(0, 1), null, Call2(0, 2)),
      Row(FastIndexedSeq(4, 1, 1), FastIndexedSeq(4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), 6, FastIndexedSeq(1, 0, 0)),
      initOpArgs = Some(FastIndexedSeq(I32(3))))
  }


  @Test def inbreeding() {
    runAggregator(
      Inbreeding(),
      TStruct("x" -> TCall(), "y" -> TFloat64()),
      FastIndexedSeq(Row(Call2(0, 0), 0d), Row(Call2(0, 1), 0.1),
        Row(Call2(0, 1), 0.2), Row(null, 0.3),
        Row(Call2(1, 1), 0.4), Row(Call2(0, 0), null)),
      Row(-1.040816, 4L, 3.02, 2L),
      FastIndexedSeq(),
      None,
      seqOpArgs = FastIndexedSeq(Ref("x", TCall()), Ref("y", TFloat64())))
  }

  @Test def infoScore() {
    runAggregator(InfoScore(), TArray(TFloat64()),
      FastIndexedSeq(FastIndexedSeq(0.3, 0.69, 0.01), FastIndexedSeq(0.3, 0.4, 0.3), null),
      Row(-0.6654567, 2))
  }

  @Test def hardyWeinberg() {
    runAggregator(HardyWeinberg(), TCall(),
      FastIndexedSeq(Call2(0, 0), Call2(0, 1), Call2(0, 1), Call2(1, 1), null),
      Row(0.571429, 0.657143))
  }

  // FIXME Max Boolean not supported by old-style MaxAggregator

  @Test def maxInt32() {
    runAggregator(Max(), TInt32(), FastIndexedSeq(), null)
    runAggregator(Max(), TInt32(), FastIndexedSeq(null), null)
    runAggregator(Max(), TInt32(), FastIndexedSeq(-2, null, 7), 7)
  }

  @Test def maxInt64() {
    runAggregator(Max(), TInt64(), FastIndexedSeq(-2L, null, 7L), 7L)
  }

  @Test def maxFloat32() {
    runAggregator(Max(), TFloat32(), FastIndexedSeq(-2.0f, null, 7.2f), 7.2f)
  }

  @Test def maxFloat64() {
    runAggregator(Max(), TFloat64(), FastIndexedSeq(-2.0, null, 7.2), 7.2)
  }

  @Test def takeInt32() {
    runAggregator(Take(), TInt32(), FastIndexedSeq(2, null, 7), FastIndexedSeq(2, null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeInt64() {
    runAggregator(Take(), TInt64(), FastIndexedSeq(2L, null, 7L), FastIndexedSeq(2L, null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat32() {
    runAggregator(Take(), TFloat32(), FastIndexedSeq(2.0f, null, 7.2f), FastIndexedSeq(2.0f, null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat64() {
    runAggregator(Take(), TFloat64(), FastIndexedSeq(2.0, null, 7.2), FastIndexedSeq(2.0, null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeCall() {
    runAggregator(Take(), TCall(), FastIndexedSeq(Call2(0, 0), null, Call2(1, 0)), FastIndexedSeq(Call2(0, 0), null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def takeString() {
    runAggregator(Take(), TString(), FastIndexedSeq("a", null, "b"), FastIndexedSeq("a", null),
      constrArgs = FastIndexedSeq(I32(2)))
  }

  @Test def testHist() {
    runAggregator(Histogram(), TFloat64(),
      FastIndexedSeq(-10.0, 0.5, 2.0, 2.5, 4.4, 4.6, 9.5, 20.0, 20.0),
      Row((0 to 10).map(_.toDouble).toFastIndexedSeq, FastIndexedSeq(1L, 0L, 2L, 0L, 2L, 0L, 0L, 0L, 0L, 1L), 1L, 2L),
      constrArgs = FastIndexedSeq(F64(0.0), F64(10.0), I32(10)))
  }

  @Test
  def ifInApplyAggOp() {
    val aggSig = AggSignature(Sum(), FastSeq(), None, FastSeq(TFloat64()))
    assertEvalsTo(
      ApplyAggOp(
        If(
          ApplyComparisonOp(NEQ(TFloat64()), Ref("a", TFloat64()), F64(10.0)),
          SeqOp(I32(0), FastSeq(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64()))),
            aggSig),
          Begin(FastSeq())),
        FastSeq(), None, aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      10.0)
  }

  @Test
  def sumMultivar() {
    val aggSig = AggSignature(Sum(), FastSeq(), None, FastSeq(TFloat64()))
    assertEvalsTo(ApplyAggOp(
      SeqOp(I32(0), FastSeq(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64()))), aggSig),
      FastSeq(), None, aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      110.0)
  }

  private[this] def assertArraySumEvalsTo[T: HailRep](
    a: IndexedSeq[Seq[T]],
    expected: Seq[T]
  ): Unit = {
    val aggSig = AggSignature(Sum(), FastSeq(), None, FastSeq(TArray(hailType[T])))
    val aggregable = a.map(Row(_))
    assertEvalsTo(
      ApplyAggOp(SeqOp(I32(0), FastSeq(Ref("a", TArray(hailType[T]))), aggSig), FastSeq(), None, aggSig),
      (aggregable, TStruct("a" -> TArray(hailType[T]))),
      expected)
  }

  @Test
  def arraySumFloat64OnEmpty(): Unit =
    assertArraySumEvalsTo[Double](
      FastIndexedSeq(),
      null
    )

  @Test
  def arraySumFloat64OnSingletonMissing(): Unit =
    assertArraySumEvalsTo[Double](
      FastIndexedSeq(null),
      null
    )

  @Test
  def arraySumFloat64OnAllMissing(): Unit =
    assertArraySumEvalsTo[Double](
      FastIndexedSeq(null, null, null),
      null
    )

  @Test
  def arraySumInt64OnEmpty(): Unit =
    assertArraySumEvalsTo[Long](
      FastIndexedSeq(),
      null
    )

  @Test
  def arraySumInt64OnSingletonMissing(): Unit =
    assertArraySumEvalsTo[Long](
      FastIndexedSeq(null),
      null
    )

  @Test
  def arraySumInt64OnAllMissing(): Unit =
    assertArraySumEvalsTo[Long](
      FastIndexedSeq(null, null, null),
      null
    )

  @Test
  def arraySumFloat64OnSmallArray(): Unit =
    assertArraySumEvalsTo(
      FastIndexedSeq(
        FastSeq(1.0, 2.0),
        FastSeq(10.0, 20.0),
        null),
      FastSeq(11.0, 22.0)
    )

  @Test
  def arraySumInt64OnSmallArray(): Unit =
    assertArraySumEvalsTo(
      FastIndexedSeq(
        FastSeq(1L, 2L),
        FastSeq(10L, 20L),
        null),
      FastSeq(11L, 22L)
    )

  @Test
  def arraySumInt64FirstElementMissing(): Unit =
    assertArraySumEvalsTo(
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
      constrArgs = FastIndexedSeq(I32(n)),
      initOpArgs = None,
      seqOpArgs = FastIndexedSeq(Ref("x", aggType), Ref("y", keyType)))
  }

  @Test def takeByNGreater() {
    assertTakeByEvalsTo(TInt32(), TInt32(), 5,
      FastIndexedSeq(Row(3, 4)),
      FastIndexedSeq(3))
  }

  @Test def takeByBooleanBoolean() {
    assertTakeByEvalsTo(TBoolean(), TBoolean(), 3,
      FastIndexedSeq(Row(false, true), Row(null, null), Row(true, false)),
      FastIndexedSeq(true, false, null))
  }

  @Test def takeByBooleanInt() {
    assertTakeByEvalsTo(TBoolean(), TInt32(), 3,
      FastIndexedSeq(Row(false, 0), Row(null, null), Row(true, 1), Row(false, 3), Row(true, null), Row(null, 2)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanLong() {
    assertTakeByEvalsTo(TBoolean(), TInt64(), 3,
      FastIndexedSeq(Row(false, 0L), Row(null, null), Row(true, 1L), Row(false, 3L), Row(true, null), Row(null, 2L)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanFloat() {
    assertTakeByEvalsTo(TBoolean(), TFloat32(), 3,
      FastIndexedSeq(Row(false, 0F), Row(null, null), Row(true, 1F), Row(false, 3F), Row(true, null), Row(null, 2F)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanDouble() {
    assertTakeByEvalsTo(TBoolean(), TFloat64(), 3,
      FastIndexedSeq(Row(false, 0D), Row(null, null), Row(true, 1D), Row(false, 3D), Row(true, null), Row(null, 2D)),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByBooleanAnnotation() {
    assertTakeByEvalsTo(TBoolean(), TString(), 3,
      FastIndexedSeq(Row(false, "a"), Row(null, null), Row(true, "b"), Row(false, "d"), Row(true, null), Row(null, "c")),
      FastIndexedSeq(false, true, null))
  }

  @Test def takeByIntBoolean() {
    assertTakeByEvalsTo(TInt32(), TBoolean(), 2,
      FastIndexedSeq(Row(3, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3))
  }

  @Test def takeByIntInt() {
    assertTakeByEvalsTo(TInt32(), TInt32(), 3,
      FastIndexedSeq(Row(3, 4), Row(null, null), Row(null, 2), Row(11, 0), Row(45, 1), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntLong() {
    assertTakeByEvalsTo(TInt32(), TInt64(), 3,
      FastIndexedSeq(Row(3, 4L), Row(null, null), Row(null, 2L), Row(11, 0L), Row(45, 1L), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntFloat() {
    assertTakeByEvalsTo(TInt32(), TFloat32(), 3,
      FastIndexedSeq(Row(3, 4F), Row(null, null), Row(null, 2F), Row(11, 0F), Row(45, 1F), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntDouble() {
    assertTakeByEvalsTo(TInt32(), TFloat64(), 3,
      FastIndexedSeq(Row(3, 4D), Row(null, null), Row(null, 2D), Row(11, 0D), Row(45, 1D), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByIntAnnotation() {
    assertTakeByEvalsTo(TInt32(), TString(), 3,
      FastIndexedSeq(Row(3, "d"), Row(null, null), Row(null, "c"), Row(11, "a"), Row(45, "b"), Row(3, null)),
      FastIndexedSeq(11, 45, null))
  }

  @Test def takeByLongBoolean() {
    assertTakeByEvalsTo(TInt64(), TBoolean(), 2,
      FastIndexedSeq(Row(3L, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3L))
  }

  @Test def takeByLongInt() {
    assertTakeByEvalsTo(TInt64(), TInt32(), 3,
      FastIndexedSeq(Row(3L, 4), Row(null, null), Row(null, 2), Row(11L, 0), Row(45L, 1), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongLong() {
    assertTakeByEvalsTo(TInt64(), TInt64(), 3,
      FastIndexedSeq(Row(3L, 4L), Row(null, null), Row(null, 2L), Row(11L, 0L), Row(45L, 1L), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongFloat() {
    assertTakeByEvalsTo(TInt64(), TFloat32(), 3,
      FastIndexedSeq(Row(3L, 4F), Row(null, null), Row(null, 2F), Row(11L, 0F), Row(45L, 1F), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongDouble() {
    assertTakeByEvalsTo(TInt64(), TFloat64(), 3,
      FastIndexedSeq(Row(3L, 4D), Row(null, null), Row(null, 2D), Row(11L, 0D), Row(45L, 1D), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByLongAnnotation() {
    assertTakeByEvalsTo(TInt64(), TString(), 3,
      FastIndexedSeq(Row(3L, "d"), Row(null, null), Row(null, "c"), Row(11L, "a"), Row(45L, "b"), Row(3L, null)),
      FastIndexedSeq(11L, 45L, null))
  }

  @Test def takeByFloatBoolean() {
    assertTakeByEvalsTo(TFloat32(), TBoolean(), 2,
      FastIndexedSeq(Row(3F, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3F))
  }

  @Test def takeByFloatInt() {
    assertTakeByEvalsTo(TFloat32(), TInt32(), 3,
      FastIndexedSeq(Row(3F, 4), Row(null, null), Row(null, 2), Row(11F, 0), Row(45F, 1), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatLong() {
    assertTakeByEvalsTo(TFloat32(), TInt64(), 3,
      FastIndexedSeq(Row(3F, 4L), Row(null, null), Row(null, 2L), Row(11F, 0L), Row(45F, 1L), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatFloat() {
    assertTakeByEvalsTo(TFloat32(), TFloat32(), 3,
      FastIndexedSeq(Row(3F, 4F), Row(null, null), Row(null, 2F), Row(11F, 0F), Row(45F, 1F), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatDouble() {
    assertTakeByEvalsTo(TFloat32(), TFloat64(), 3,
      FastIndexedSeq(Row(3F, 4D), Row(null, null), Row(null, 2D), Row(11F, 0D), Row(45F, 1D), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByFloatAnnotation() {
    assertTakeByEvalsTo(TFloat32(), TString(), 3,
      FastIndexedSeq(Row(3F, "d"), Row(null, null), Row(null, "c"), Row(11F, "a"), Row(45F, "b"), Row(3F, null)),
      FastIndexedSeq(11F, 45F, null))
  }

  @Test def takeByDoubleBoolean() {
    assertTakeByEvalsTo(TFloat64(), TBoolean(), 2,
      FastIndexedSeq(Row(3D, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3D))
  }

  @Test def takeByDoubleInt() {
    assertTakeByEvalsTo(TFloat64(), TInt32(), 3,
      FastIndexedSeq(Row(3D, 4), Row(null, null), Row(null, 2), Row(11D, 0), Row(45D, 1), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleLong() {
    assertTakeByEvalsTo(TFloat64(), TInt64(), 3,
      FastIndexedSeq(Row(3D, 4L), Row(null, null), Row(null, 2L), Row(11D, 0L), Row(45D, 1L), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleFloat() {
    assertTakeByEvalsTo(TFloat64(), TFloat32(), 3,
      FastIndexedSeq(Row(3D, 4F), Row(null, null), Row(null, 2F), Row(11D, 0F), Row(45D, 1F), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleDouble() {
    assertTakeByEvalsTo(TFloat64(), TFloat64(), 3,
      FastIndexedSeq(Row(3D, 4D), Row(null, null), Row(null, 2D), Row(11D, 0D), Row(45D, 1D), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByDoubleAnnotation() {
    assertTakeByEvalsTo(TFloat64(), TString(), 3,
      FastIndexedSeq(Row(3D, "d"), Row(null, null), Row(null, "c"), Row(11D, "a"), Row(45D, "b"), Row(3D, null)),
      FastIndexedSeq(11D, 45D, null))
  }

  @Test def takeByAnnotationBoolean() {
    assertTakeByEvalsTo(TString(), TBoolean(), 2,
      FastIndexedSeq(Row("hello", true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, "hello"))
  }

  @Test def takeByAnnotationInt() {
    assertTakeByEvalsTo(TString(), TInt32(), 3,
      FastIndexedSeq(Row("a", 4), Row(null, null), Row(null, 2), Row("b", 0), Row("c", 1), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationLong() {
    assertTakeByEvalsTo(TString(), TInt64(), 3,
      FastIndexedSeq(Row("a", 4L), Row(null, null), Row(null, 2L), Row("b", 0L), Row("c", 1L), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationFloat() {
    assertTakeByEvalsTo(TString(), TFloat32(), 3,
      FastIndexedSeq(Row("a", 4F), Row(null, null), Row(null, 2F), Row("b", 0F), Row("c", 1F), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationDouble() {
    assertTakeByEvalsTo(TString(), TFloat64(), 3,
      FastIndexedSeq(Row("a", 4D), Row(null, null), Row(null, 2D), Row("b", 0D), Row("c", 1D), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByAnnotationAnnotation() {
    assertTakeByEvalsTo(TString(), TString(), 3,
      FastIndexedSeq(Row("a", "d"), Row(null, null), Row(null, "c"), Row("b", "a"), Row("c", "b"), Row("d", null)),
      FastIndexedSeq("b", "c", null))
  }

  @Test def takeByCallLong() {
    assertTakeByEvalsTo(TCall(), TInt64(), 3,
      FastIndexedSeq(Row(Call2(0, 0), 4L), Row(null, null), Row(null, 2L), Row(Call2(0, 1), 0L), Row(Call2(1, 1), 1L), Row(Call2(0, 2), null)),
      FastIndexedSeq(Call2(0, 1), Call2(1, 1), null))
  }
}

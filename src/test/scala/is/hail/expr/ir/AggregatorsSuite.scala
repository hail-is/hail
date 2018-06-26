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
  def runAggregator(op: AggOp, t: Type, a: IndexedSeq[Any], expected: Any, args: IndexedSeq[IR] = FastIndexedSeq(),
    initOpArgs: Option[IndexedSeq[IR]] = None, seqOpArgs: IndexedSeq[IR] = FastIndexedSeq()) {
    val aggSig = AggSignature(op, t, args.map(_.typ), initOpArgs.map(_.map(_.typ)), seqOpArgs.map(_.typ))
    assertEvalsTo(ApplyAggOp(
      SeqOp(Ref("x", t), I32(0), aggSig, seqOpArgs),
      args, initOpArgs, aggSig),
      (a.map(i => Row(i)), TStruct("x" -> t)),
      expected)
  }

  def runAggregator(
    op: AggOp,
    t: TStruct,
    a: IndexedSeq[Row],
    expected: Any,
    aggIR: IR,
    args: IndexedSeq[IR],
    initOpArgs: Option[IndexedSeq[IR]],
    seqOpArgs: IndexedSeq[IR]) {
    val aggSig = AggSignature(op, aggIR.typ, args.map(_.typ), initOpArgs.map(_.map(_.typ)), seqOpArgs.map(_.typ))
    assertEvalsTo(ApplyAggOp(
      SeqOp(aggIR, I32(0), aggSig, seqOpArgs),
      args, initOpArgs, aggSig),
      (a, t),
      expected)
  }

  @Test def sumFloat64() {
    runAggregator(Sum(), TFloat64(), (0 to 100).map(_.toDouble), 5050.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(), 0.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(42.0), 42.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(null, 42.0, null), 42.0)
    runAggregator(Sum(), TFloat64(), FastIndexedSeq(null, null, null), 0.0)
  }

  def sumInt64() {
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

  @Test def collectStruct(): Unit = {
    runAggregator(Collect(),
      TStruct("a" -> TInt32(), "b" -> TBoolean()),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null),
      FastIndexedSeq(Row(5, true), Row(3, false), null, Row(0, false), null))
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
      Ref("x", TCall()),
      FastIndexedSeq(),
      None,
      seqOpArgs = FastIndexedSeq(Ref("y", TFloat64())))
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
      args = FastIndexedSeq(I32(2)))
  }

  @Test def takeInt64() {
    runAggregator(Take(), TInt64(), FastIndexedSeq(2L, null, 7L), FastIndexedSeq(2L, null),
      args = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat32() {
    runAggregator(Take(), TFloat32(), FastIndexedSeq(2.0f, null, 7.2f), FastIndexedSeq(2.0f, null),
      args = FastIndexedSeq(I32(2)))
  }

  @Test def takeFloat64() {
    runAggregator(Take(), TFloat64(), FastIndexedSeq(2.0, null, 7.2), FastIndexedSeq(2.0, null),
      args = FastIndexedSeq(I32(2)))
  }

  @Test def takeCall() {
    runAggregator(Take(), TCall(), FastIndexedSeq(Call2(0, 0), null, Call2(1, 0)), FastIndexedSeq(Call2(0, 0), null),
      args = FastIndexedSeq(I32(2)))
  }

  @Test def takeString() {
    runAggregator(Take(), TString(), FastIndexedSeq("a", null, "b"), FastIndexedSeq("a", null),
      args = FastIndexedSeq(I32(2)))
  }

  @Test def testHist() {
    runAggregator(Histogram(), TFloat64(),
      FastIndexedSeq(-10.0, 0.5, 2.0, 2.5, 4.4, 4.6, 9.5, 20.0, 20.0),
      Row((0 to 10).map(_.toDouble).toFastIndexedSeq, FastIndexedSeq(1L, 0L, 2L, 0L, 2L, 0L, 0L, 0L, 0L, 1L), 1L, 2L),
      args = FastIndexedSeq(F64(0.0), F64(10.0), I32(10)))
  }

  @Test
  def ifInApplyAggOp() {
    val aggSig = AggSignature(Sum(), TFloat64(), FastSeq(), None, FastSeq())
    assertEvalsTo(
      ApplyAggOp(
        If(
          ApplyComparisonOp(NEQ(TFloat64()), Ref("a", TFloat64()), F64(10.0)),
          SeqOp(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64())),
            I32(0), aggSig),
          Begin(FastSeq())),
        FastSeq(), None, aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      10.0)
  }

  @Test
  def sumMultivar() {
    val aggSig = AggSignature(Sum(), TFloat64(), FastSeq(), None, FastSeq())
    assertEvalsTo(ApplyAggOp(
      SeqOp(ApplyBinaryPrimOp(Multiply(), Ref("a", TFloat64()), Ref("b", TFloat64())), I32(0), aggSig),
      FastSeq(), None, aggSig),
      (FastIndexedSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)), TStruct("a" -> TFloat64(), "b" -> TFloat64())),
      110.0)
  }

  private[this] def assertArraySumEvalsTo[T: HailRep](
    a: IndexedSeq[Seq[T]],
    expected: Seq[T]
  ): Unit = {
    val aggSig = AggSignature(Sum(), TArray(hailType[T]), FastSeq(), None, FastSeq())
    val aggregable = a.map(Row(_))
    assertEvalsTo(
      ApplyAggOp(SeqOp(Ref("a", TArray(hailType[T])), I32(0), aggSig), FastSeq(), None, aggSig),
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
      Ref("x", aggType),
      args = FastIndexedSeq(I32(n)),
      initOpArgs = None,
      seqOpArgs = FastIndexedSeq(Ref("y", keyType)))
  }

  @Test def takeByNGreater() {
    assertTakeByEvalsTo(TFloat64(), TFloat64(), 5,
      FastIndexedSeq(Row(3D, 4D)),
      FastIndexedSeq(3D))
  }

  @Test def takeByBooleanBoolean() {
    assertTakeByEvalsTo(TBoolean(), TBoolean(), 3,
      FastIndexedSeq(Row(false, true), Row(null, null), Row(true, false)),
      FastIndexedSeq(true, false, null))
  }

  @Test def takeByBooleanCall() {
    assertTakeByEvalsTo(TBoolean(), TCall(), 3,
      FastIndexedSeq(Row(false, Call2(0, 0)), Row(null, null), Row(true, Call2(0, 1))),
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

  @Test def takeByCallBoolean() {
    assertTakeByEvalsTo(TCall(), TBoolean(), 2,
      FastIndexedSeq(Row(Call2(0, 0), true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, Call2(0, 0)))
  }

  @Test def takeByCallCall() {
    assertTakeByEvalsTo(TCall(), TCall(), 3,
      FastIndexedSeq(Row(Call2(0, 0), Call2(1, 2)), Row(null, null), Row(null, Call2(0, 2)), Row(Call2(0, 1), Call2(0, 0)), Row(Call2(1, 1), Call2(0, 1)), Row(Call2(2, 2), null)),
      FastIndexedSeq(Call2(0, 1), Call2(1, 1), null))
  }

  @Test def takeByCallDouble() {
    assertTakeByEvalsTo(TCall(), TFloat64(), 3,
      FastIndexedSeq(Row(Call2(0, 0), 4D), Row(null, null), Row(null, 2D), Row(Call2(0, 1), 0D), Row(Call2(1, 1), 1D), Row(Call2(2, 2), null)),
      FastIndexedSeq(Call2(0, 1), Call2(1, 1), null))
  }

  @Test def takeByCallAnnotation() {
    assertTakeByEvalsTo(TCall(), TString(), 3,
      FastIndexedSeq(Row(Call2(0, 0), "d"), Row(null, null), Row(null, "c"), Row(Call2(0, 1), "a"), Row(Call2(1, 1), "b"), Row(Call2(2, 2), null)),
      FastIndexedSeq(Call2(0, 1), Call2(1, 1), null))
  }

  @Test def takeByDoubleBoolean() {
    assertTakeByEvalsTo(TFloat64(), TBoolean(), 2,
      FastIndexedSeq(Row(3D, true), Row(null, null), Row(null, false)),
      FastIndexedSeq(null, 3D))
  }

  @Test def takeByDoubleCall() {
    assertTakeByEvalsTo(TFloat64(), TCall(), 3,
      FastIndexedSeq(Row(3D, Call2(0, 2)), Row(null, null), Row(null, Call2(1, 1)), Row(11D, Call2(0, 0)), Row(45D, Call2(0, 1)), Row(3D, null)),
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

  @Test def takeByAnnotationCall() {
    assertTakeByEvalsTo(TString(), TCall(), 3,
      FastIndexedSeq(Row("a", Call2(0, 2)), Row(null, null), Row(null, Call2(1, 1)), Row("b", Call2(0, 0)), Row("c", Call2(0, 1)), Row("d", null)),
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
}

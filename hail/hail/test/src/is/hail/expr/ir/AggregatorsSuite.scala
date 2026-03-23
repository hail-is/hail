package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.DeprecatedIRBuilder._
import is.hail.expr.ir.defs.{
  AggFilter, AggGroupBy, ApplyAggOp, ApplyBinaryPrimOp, ArrayRef, Cast, GetField, I32, InsertFields,
  MakeStruct, MakeTuple, Ref, Str, StreamAgg, StreamAggScan, StreamRange, TableAggregate, ToArray,
  ToStream,
}
import is.hail.expr.ir.lowering.{DArrayLowering, LowerTableIR}
import is.hail.types.virtual._
import is.hail.variant.Call2

import org.apache.spark.sql.Row

class AggregatorsSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

  def runAggregator(
    op: AggOp,
    aggType: TStruct,
    agg: IndexedSeq[Row],
    expected: Any,
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR],
  ): Unit =
    assertEvalsTo(
      ApplyAggOp(initOpArgs, seqOpArgs, op),
      (agg, aggType),
      expected,
    )

  def runAggregator(
    op: AggOp,
    t: Type,
    a: IndexedSeq[Any],
    expected: Any,
    initOpArgs: IndexedSeq[IR] = FastSeq(),
  ): Unit = {
    runAggregator(
      op,
      TStruct("x" -> t),
      a.map(i => Row(i)),
      expected,
      initOpArgs,
      seqOpArgs = FastSeq(Ref(Name("x"), t)),
    )
  }

  test("nestedAgg") {
    val agg = ToArray(mapIR(StreamRange(0, 10, 1))(_ => ApplyAggOp(Count())()))
    assertEvalsTo(
      agg,
      (FastSeq(1, 2).map(i => Row(i)), TStruct("x" -> TInt32)),
      IndexedSeq.fill(10)(2L),
    )
  }

  test("sumFloat64") {
    runAggregator(Sum(), TFloat64, (0 to 100).map(_.toDouble), 5050.0)
    runAggregator(Sum(), TFloat64, FastSeq(), 0.0)
    runAggregator(Sum(), TFloat64, FastSeq(42.0), 42.0)
    runAggregator(Sum(), TFloat64, FastSeq(null, 42.0, null), 42.0)
    runAggregator(Sum(), TFloat64, FastSeq(null, null, null), 0.0)
  }

  test("sumInt64") {
    runAggregator(Sum(), TInt64, FastSeq(-1L, 2L, 3L), 4L)
  }

  test("collectBoolean") {
    runAggregator(
      Collect(),
      TBoolean,
      FastSeq(true, false, null, true, false),
      FastSeq(true, false, null, true, false),
    )
  }

  test("collectInt") {
    runAggregator(Collect(), TInt32, FastSeq(10, null, 5), FastSeq(10, null, 5))
  }

  test("collectLong") {
    runAggregator(Collect(), TInt64, FastSeq(10L, null, 5L), FastSeq(10L, null, 5L))
  }

  test("collectFloat") {
    runAggregator(Collect(), TFloat32, FastSeq(10f, null, 5f), FastSeq(10f, null, 5f))
  }

  test("collectDouble") {
    runAggregator(Collect(), TFloat64, FastSeq(10d, null, 5d), FastSeq(10d, null, 5d))
  }

  test("collectString") {
    runAggregator(Collect(), TString, FastSeq("hello", null, "foo"), FastSeq("hello", null, "foo"))
  }

  test("collectArray") {
    runAggregator(
      Collect(),
      TArray(TInt32),
      FastSeq(FastSeq(1, 2, 3), null, FastSeq()),
      FastSeq(FastSeq(1, 2, 3), null, FastSeq()),
    )
  }

  test("collectStruct") {
    runAggregator(
      Collect(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastSeq(Row(5, true), Row(3, false), null, Row(0, false), null),
      FastSeq(Row(5, true), Row(3, false), null, Row(0, false), null),
    )
  }

  test("count") {
    runAggregator(
      Count(),
      TStruct("x" -> TString),
      FastSeq(Row("hello"), Row("foo"), Row("a"), Row(null), Row("b"), Row(null), Row("c")),
      7L,
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )
  }

  test("collectAsSetBoolean") {
    runAggregator(
      CollectAsSet(),
      TBoolean,
      FastSeq(true, false, null, true, false),
      Set(true, false, null),
    )
    runAggregator(CollectAsSet(), TBoolean, FastSeq(true, null, true), Set(true, null))
  }

  test("collectAsSetNumeric") {
    runAggregator(CollectAsSet(), TInt32, FastSeq(10, null, 5, 5, null), Set(10, null, 5))
    runAggregator(CollectAsSet(), TInt64, FastSeq(10L, null, 5L, 5L, null), Set(10L, null, 5L))
    runAggregator(CollectAsSet(), TFloat32, FastSeq(10f, null, 5f, 5f, null), Set(10f, null, 5f))
    runAggregator(CollectAsSet(), TFloat64, FastSeq(10d, null, 5d, 5d, null), Set(10d, null, 5d))
  }

  test("collectAsSetString") {
    runAggregator(
      CollectAsSet(),
      TString,
      FastSeq("hello", null, "foo", null, "foo"),
      Set("hello", null, "foo"),
    )
  }

  test("collectAsSetArray") {
    val inputCollection = FastSeq(FastSeq(1, 2, 3), null, FastSeq(), null, FastSeq(1, 2, 3))
    val expected = Set(FastSeq(1, 2, 3), null, FastSeq())
    runAggregator(CollectAsSet(), TArray(TInt32), inputCollection, expected)
  }

  test("collectAsSetStruct") {
    runAggregator(
      CollectAsSet(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastSeq(Row(5, true), Row(3, false), null, Row(0, false), null, Row(5, true)),
      Set(Row(5, true), Row(3, false), null, Row(0, false)),
    )
  }

  test("callStats") {
    runAggregator(
      CallStats(),
      TCall,
      FastSeq(Call2(0, 0), Call2(0, 1), null, Call2(0, 2)),
      Row(FastSeq(4, 1, 1), FastSeq(4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), 6, FastSeq(1, 0, 0)),
      initOpArgs = FastSeq(I32(3)),
    )
  }

  // FIXME Max Boolean not supported by old-style MaxAggregator

  test("maxInt32") {
    runAggregator(Max(), TInt32, FastSeq(), null)
    runAggregator(Max(), TInt32, FastSeq(null), null)
    runAggregator(Max(), TInt32, FastSeq(-2, null, 7), 7)
  }

  test("maxInt64") {
    runAggregator(Max(), TInt64, FastSeq(-2L, null, 7L), 7L)
  }

  test("maxFloat32") {
    runAggregator(Max(), TFloat32, FastSeq(-2.0f, null, 7.2f), 7.2f)
  }

  test("maxFloat64") {
    runAggregator(Max(), TFloat64, FastSeq(-2.0, null, 7.2), 7.2)
  }

  test("takeInt32") {
    runAggregator(
      Take(),
      TInt32,
      FastSeq(2, null, 7),
      FastSeq(2, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("takeInt64") {
    runAggregator(
      Take(),
      TInt64,
      FastSeq(2L, null, 7L),
      FastSeq(2L, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("takeFloat32") {
    runAggregator(
      Take(),
      TFloat32,
      FastSeq(2.0f, null, 7.2f),
      FastSeq(2.0f, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("takeFloat64") {
    runAggregator(
      Take(),
      TFloat64,
      FastSeq(2.0, null, 7.2),
      FastSeq(2.0, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("takeCall") {
    runAggregator(
      Take(),
      TCall,
      FastSeq(Call2(0, 0), null, Call2(1, 0)),
      FastSeq(Call2(0, 0), null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("takeString") {
    runAggregator(
      Take(),
      TString,
      FastSeq("a", null, "b"),
      FastSeq("a", null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  test("sumMultivar") {
    assertEvalsTo(
      ApplyAggOp(
        FastSeq(),
        FastSeq(ApplyBinaryPrimOp(Multiply(), Ref(Name("a"), TFloat64), Ref(Name("b"), TFloat64))),
        Sum(),
      ),
      (
        FastSeq(Row(1.0, 10.0), Row(10.0, 10.0), Row(null, 10.0)),
        TStruct("a" -> TFloat64, "b" -> TFloat64),
      ),
      110.0,
    )
  }

  private[this] def assertArraySumEvalsTo[T](
    eltType: Type,
    a: IndexedSeq[Seq[T]],
    expected: Seq[T],
  ): Unit = {
    val aggregable = a.map(Row(_))
    val structType = TStruct("foo" -> TArray(eltType))

    assertEvalsTo(
      aggArrayPerElement(Ref(Name("foo"), TArray(eltType))) { (elt, _) =>
        ApplyAggOp(FastSeq(), FastSeq(elt), Sum())
      },
      (aggregable, structType),
      expected,
    )
  }

  test("arraySumFloat64OnEmpty") {
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(),
      null,
    )
  }

  test("arraySumFloat64OnSingletonMissing") {
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(null),
      null,
    )
  }

  test("arraySumFloat64OnAllMissing") {
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(null, null, null),
      null,
    )
  }

  test("arraySumInt64OnEmpty") {
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(),
      null,
    )
  }

  test("arraySumInt64OnSingletonMissing") {
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(null),
      null,
    )
  }

  test("arraySumInt64OnAllMissing") {
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(null, null, null),
      null,
    )
  }

  test("arraySumFloat64OnSmallArray") {
    assertArraySumEvalsTo(
      TFloat64,
      FastSeq(
        FastSeq(1.0, 2.0),
        FastSeq(10.0, 20.0),
        null,
      ),
      FastSeq(11.0, 22.0),
    )
  }

  test("arraySumInt64OnSmallArray") {
    assertArraySumEvalsTo(
      TInt64,
      FastSeq(
        FastSeq(1L, 2L),
        FastSeq(10L, 20L),
        null,
      ),
      FastSeq(11L, 22L),
    )
  }

  test("arraySumInt64FirstElementMissing") {
    assertArraySumEvalsTo(
      TInt64,
      FastSeq(
        null,
        FastSeq(1L, 33L),
        FastSeq(42L, 3L),
      ),
      FastSeq(43L, 36L),
    )
  }

  private[this] def assertTakeByEvalsTo(
    aggType: Type,
    keyType: Type,
    n: Int,
    a: IndexedSeq[Row],
    expected: IndexedSeq[Any],
  ): Unit = {
    runAggregator(
      TakeBy(),
      TStruct("x" -> aggType, "y" -> keyType),
      a,
      expected,
      initOpArgs = FastSeq(I32(n)),
      seqOpArgs = FastSeq(Ref(Name("x"), aggType), Ref(Name("y"), keyType)),
    )
  }

  test("takeByNGreater") {
    assertTakeByEvalsTo(TInt32, TInt32, 5, FastSeq(Row(3, 4)), FastSeq(3))
  }

  test("takeByBooleanBoolean") {
    assertTakeByEvalsTo(
      TBoolean,
      TBoolean,
      3,
      FastSeq(Row(false, true), Row(null, null), Row(true, false)),
      FastSeq(true, false, null),
    )
  }

  test("takeByBooleanInt") {
    assertTakeByEvalsTo(
      TBoolean,
      TInt32,
      3,
      FastSeq(
        Row(false, 0),
        Row(null, null),
        Row(true, 1),
        Row(false, 3),
        Row(true, null),
        Row(null, 2),
      ),
      FastSeq(false, true, null),
    )
  }

  test("takeByBooleanLong") {
    assertTakeByEvalsTo(
      TBoolean,
      TInt64,
      3,
      FastSeq(
        Row(false, 0L),
        Row(null, null),
        Row(true, 1L),
        Row(false, 3L),
        Row(true, null),
        Row(null, 2L),
      ),
      FastSeq(false, true, null),
    )
  }

  test("takeByBooleanFloat") {
    assertTakeByEvalsTo(
      TBoolean,
      TFloat32,
      3,
      FastSeq(
        Row(false, 0f),
        Row(null, null),
        Row(true, 1f),
        Row(false, 3f),
        Row(true, null),
        Row(null, 2f),
      ),
      FastSeq(false, true, null),
    )
  }

  test("takeByBooleanDouble") {
    assertTakeByEvalsTo(
      TBoolean,
      TFloat64,
      3,
      FastSeq(
        Row(false, 0d),
        Row(null, null),
        Row(true, 1d),
        Row(false, 3d),
        Row(true, null),
        Row(null, 2d),
      ),
      FastSeq(false, true, null),
    )
  }

  test("takeByBooleanAnnotation") {
    assertTakeByEvalsTo(
      TBoolean,
      TString,
      3,
      FastSeq(
        Row(false, "a"),
        Row(null, null),
        Row(true, "b"),
        Row(false, "d"),
        Row(true, null),
        Row(null, "c"),
      ),
      FastSeq(false, true, null),
    )
  }

  test("takeByIntBoolean") {
    assertTakeByEvalsTo(
      TInt32,
      TBoolean,
      2,
      FastSeq(Row(3, true), Row(null, null), Row(null, false)),
      FastSeq(null, 3),
    )
  }

  test("takeByIntInt") {
    assertTakeByEvalsTo(
      TInt32,
      TInt32,
      3,
      FastSeq(Row(3, 4), Row(null, null), Row(null, 2), Row(11, 0), Row(45, 1), Row(3, null)),
      FastSeq(11, 45, null),
    )
  }

  test("takeByIntLong") {
    assertTakeByEvalsTo(
      TInt32,
      TInt64,
      3,
      FastSeq(Row(3, 4L), Row(null, null), Row(null, 2L), Row(11, 0L), Row(45, 1L), Row(3, null)),
      FastSeq(11, 45, null),
    )
  }

  test("takeByIntFloat") {
    assertTakeByEvalsTo(
      TInt32,
      TFloat32,
      3,
      FastSeq(Row(3, 4f), Row(null, null), Row(null, 2f), Row(11, 0f), Row(45, 1f), Row(3, null)),
      FastSeq(11, 45, null),
    )
  }

  test("takeByIntDouble") {
    assertTakeByEvalsTo(
      TInt32,
      TFloat64,
      3,
      FastSeq(Row(3, 4d), Row(null, null), Row(null, 2d), Row(11, 0d), Row(45, 1d), Row(3, null)),
      FastSeq(11, 45, null),
    )
  }

  test("takeByIntAnnotation") {
    assertTakeByEvalsTo(
      TInt32,
      TString,
      3,
      FastSeq(
        Row(3, "d"),
        Row(null, null),
        Row(null, "c"),
        Row(11, "a"),
        Row(45, "b"),
        Row(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  test("takeByLongBoolean") {
    assertTakeByEvalsTo(
      TInt64,
      TBoolean,
      2,
      FastSeq(Row(3L, true), Row(null, null), Row(null, false)),
      FastSeq(null, 3L),
    )
  }

  test("takeByLongInt") {
    assertTakeByEvalsTo(
      TInt64,
      TInt32,
      3,
      FastSeq(Row(3L, 4), Row(null, null), Row(null, 2), Row(11L, 0), Row(45L, 1), Row(3L, null)),
      FastSeq(11L, 45L, null),
    )
  }

  test("takeByLongLong") {
    assertTakeByEvalsTo(
      TInt64,
      TInt64,
      3,
      FastSeq(
        Row(3L, 4L),
        Row(null, null),
        Row(null, 2L),
        Row(11L, 0L),
        Row(45L, 1L),
        Row(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  test("takeByLongFloat") {
    assertTakeByEvalsTo(
      TInt64,
      TFloat32,
      3,
      FastSeq(
        Row(3L, 4f),
        Row(null, null),
        Row(null, 2f),
        Row(11L, 0f),
        Row(45L, 1f),
        Row(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  test("takeByLongDouble") {
    assertTakeByEvalsTo(
      TInt64,
      TFloat64,
      3,
      FastSeq(
        Row(3L, 4d),
        Row(null, null),
        Row(null, 2d),
        Row(11L, 0d),
        Row(45L, 1d),
        Row(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  test("takeByLongAnnotation") {
    assertTakeByEvalsTo(
      TInt64,
      TString,
      3,
      FastSeq(
        Row(3L, "d"),
        Row(null, null),
        Row(null, "c"),
        Row(11L, "a"),
        Row(45L, "b"),
        Row(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  test("takeByFloatBoolean") {
    assertTakeByEvalsTo(
      TFloat32,
      TBoolean,
      2,
      FastSeq(Row(3f, true), Row(null, null), Row(null, false)),
      FastSeq(null, 3f),
    )
  }

  test("takeByFloatInt") {
    assertTakeByEvalsTo(
      TFloat32,
      TInt32,
      3,
      FastSeq(Row(3f, 4), Row(null, null), Row(null, 2), Row(11f, 0), Row(45f, 1), Row(3f, null)),
      FastSeq(11f, 45f, null),
    )
  }

  test("takeByFloatLong") {
    assertTakeByEvalsTo(
      TFloat32,
      TInt64,
      3,
      FastSeq(
        Row(3f, 4L),
        Row(null, null),
        Row(null, 2L),
        Row(11f, 0L),
        Row(45f, 1L),
        Row(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  test("takeByFloatFloat") {
    assertTakeByEvalsTo(
      TFloat32,
      TFloat32,
      3,
      FastSeq(
        Row(3f, 4f),
        Row(null, null),
        Row(null, 2f),
        Row(11f, 0f),
        Row(45f, 1f),
        Row(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  test("takeByFloatDouble") {
    assertTakeByEvalsTo(
      TFloat32,
      TFloat64,
      3,
      FastSeq(
        Row(3f, 4d),
        Row(null, null),
        Row(null, 2d),
        Row(11f, 0d),
        Row(45f, 1d),
        Row(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  test("takeByFloatAnnotation") {
    assertTakeByEvalsTo(
      TFloat32,
      TString,
      3,
      FastSeq(
        Row(3f, "d"),
        Row(null, null),
        Row(null, "c"),
        Row(11f, "a"),
        Row(45f, "b"),
        Row(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  test("takeByDoubleBoolean") {
    assertTakeByEvalsTo(
      TFloat64,
      TBoolean,
      2,
      FastSeq(Row(3d, true), Row(null, null), Row(null, false)),
      FastSeq(null, 3d),
    )
  }

  test("takeByDoubleInt") {
    assertTakeByEvalsTo(
      TFloat64,
      TInt32,
      3,
      FastSeq(Row(3d, 4), Row(null, null), Row(null, 2), Row(11d, 0), Row(45d, 1), Row(3d, null)),
      FastSeq(11d, 45d, null),
    )
  }

  test("takeByDoubleLong") {
    assertTakeByEvalsTo(
      TFloat64,
      TInt64,
      3,
      FastSeq(
        Row(3d, 4L),
        Row(null, null),
        Row(null, 2L),
        Row(11d, 0L),
        Row(45d, 1L),
        Row(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  test("takeByDoubleFloat") {
    assertTakeByEvalsTo(
      TFloat64,
      TFloat32,
      3,
      FastSeq(
        Row(3d, 4f),
        Row(null, null),
        Row(null, 2f),
        Row(11d, 0f),
        Row(45d, 1f),
        Row(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  test("takeByDoubleDouble") {
    assertTakeByEvalsTo(
      TFloat64,
      TFloat64,
      3,
      FastSeq(
        Row(3d, 4d),
        Row(null, null),
        Row(null, 2d),
        Row(11d, 0d),
        Row(45d, 1d),
        Row(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  test("takeByDoubleAnnotation") {
    assertTakeByEvalsTo(
      TFloat64,
      TString,
      3,
      FastSeq(
        Row(3d, "d"),
        Row(null, null),
        Row(null, "c"),
        Row(11d, "a"),
        Row(45d, "b"),
        Row(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  test("takeByAnnotationBoolean") {
    assertTakeByEvalsTo(
      TString,
      TBoolean,
      2,
      FastSeq(Row("hello", true), Row(null, null), Row(null, false)),
      FastSeq(null, "hello"),
    )
  }

  test("takeByAnnotationInt") {
    assertTakeByEvalsTo(
      TString,
      TInt32,
      3,
      FastSeq(Row("a", 4), Row(null, null), Row(null, 2), Row("b", 0), Row("c", 1), Row("d", null)),
      FastSeq("b", "c", null),
    )
  }

  test("takeByAnnotationLong") {
    assertTakeByEvalsTo(
      TString,
      TInt64,
      3,
      FastSeq(
        Row("a", 4L),
        Row(null, null),
        Row(null, 2L),
        Row("b", 0L),
        Row("c", 1L),
        Row("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  test("takeByAnnotationFloat") {
    assertTakeByEvalsTo(
      TString,
      TFloat32,
      3,
      FastSeq(
        Row("a", 4f),
        Row(null, null),
        Row(null, 2f),
        Row("b", 0f),
        Row("c", 1f),
        Row("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  test("takeByAnnotationDouble") {
    assertTakeByEvalsTo(
      TString,
      TFloat64,
      3,
      FastSeq(
        Row("a", 4d),
        Row(null, null),
        Row(null, 2d),
        Row("b", 0d),
        Row("c", 1d),
        Row("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  test("takeByAnnotationAnnotation") {
    assertTakeByEvalsTo(
      TString,
      TString,
      3,
      FastSeq(
        Row("a", "d"),
        Row(null, null),
        Row(null, "c"),
        Row("b", "a"),
        Row("c", "b"),
        Row("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  test("takeByCallLong") {
    assertTakeByEvalsTo(
      TCall,
      TInt64,
      3,
      FastSeq(
        Row(Call2(0, 0), 4L),
        Row(null, null),
        Row(null, 2L),
        Row(Call2(0, 1), 0L),
        Row(Call2(1, 1), 1L),
        Row(Call2(0, 2), null),
      ),
      FastSeq(Call2(0, 1), Call2(1, 1), null),
    )
  }

  def runKeyedAggregator(
    op: AggOp,
    key: IR,
    aggType: TStruct,
    agg: IndexedSeq[Row],
    expected: Any,
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR],
  ): Unit = {
    assertEvalsTo(
      AggGroupBy(
        key,
        ApplyAggOp(initOpArgs, seqOpArgs, op),
        false,
      ),
      (agg, aggType),
      expected,
    )
  }

  test("keyedCount") {
    runKeyedAggregator(
      Count(),
      Ref(Name("k"), TInt32),
      TStruct("k" -> TInt32),
      FastSeq(Row(1), Row(2), Row(3), Row(1), Row(1), Row(null), Row(null)),
      Map(1 -> 3L, 2 -> 1L, 3 -> 1L, (null, 2L)),
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )

    runKeyedAggregator(
      Count(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean),
      FastSeq(Row(true), Row(true), Row(true), Row(false), Row(false), Row(null), Row(null)),
      Map(true -> 3L, false -> 2L, (null, 2L)),
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )

    // test struct as key
    runKeyedAggregator(
      Count(),
      Ref(Name("k"), TStruct("a" -> TBoolean)),
      TStruct("k" -> TStruct("a" -> TBoolean)),
      FastSeq(
        Row(Row(true)),
        Row(Row(true)),
        Row(Row(true)),
        Row(Row(false)),
        Row(Row(false)),
        Row(Row(null)),
        Row(Row(null)),
      ),
      Map(Row(true) -> 3L, Row(false) -> 2L, (Row(null), 2L)),
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )
  }

  test("keyedCollect") {
    runKeyedAggregator(
      Collect(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean, "v" -> TInt32),
      FastSeq(
        Row(true, 5),
        Row(true, 3),
        Row(true, null),
        Row(false, 0),
        Row(false, null),
        Row(null, null),
        Row(null, 2),
      ),
      Map(true -> FastSeq(5, 3, null), false -> FastSeq(0, null), (null, FastSeq(null, 2))),
      FastSeq(),
      FastSeq(Ref(Name("v"), TInt32)),
    )
  }

  test("keyedCallStats") {
    runKeyedAggregator(
      CallStats(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean, "v" -> TCall),
      FastSeq(
        Row(true, null),
        Row(true, Call2(0, 1)),
        Row(true, Call2(0, 1)),
        Row(false, null),
        Row(false, Call2(0, 0)),
        Row(false, Call2(1, 1)),
      ),
      Map(
        true -> Row(FastSeq(2, 2), FastSeq(0.5, 0.5), 4, FastSeq(0, 0)),
        false -> Row(FastSeq(2, 2), FastSeq(0.5, 0.5), 4, FastSeq(1, 1)),
      ),
      FastSeq(I32(2)),
      FastSeq(Ref(Name("v"), TCall)),
    )
  }

  test("keyedTakeBy") {
    runKeyedAggregator(
      TakeBy(),
      Ref(Name("k"), TString),
      TStruct("k" -> TString, "x" -> TFloat64, "y" -> TInt32),
      FastSeq(
        Row("case", 0.2, 5),
        Row("control", 0.4, 0),
        Row(null, 1.0, 3),
        Row("control", 0.0, 2),
        Row("case", 0.3, 6),
        Row("control", 0.5, 1),
      ),
      Map("case" -> FastSeq(0.2, 0.3), "control" -> FastSeq(0.4, 0.5), (null, FastSeq(1.0))),
      FastSeq(I32(2)),
      FastSeq(Ref(Name("x"), TFloat64), Ref(Name("y"), TInt32)),
    )
  }

  test("keyedKeyedCollect") {
    val agg =
      FastSeq(Row("EUR", true, 1), Row("EUR", false, 2), Row("AFR", true, 3), Row("AFR", null, 4))
    val aggType = TStruct("k1" -> TString, "k2" -> TBoolean, "x" -> TInt32)
    val expected: Map[String, Map[Any, Seq[Int]]] = Map(
      "EUR" -> Map(true -> FastSeq(1), false -> FastSeq(2)),
      "AFR" -> Map(true -> FastSeq(3), (null, FastSeq(4))),
    )
    assertEvalsTo(
      AggGroupBy(
        Ref(Name("k1"), TString),
        AggGroupBy(
          Ref(Name("k2"), TBoolean),
          ApplyAggOp(FastSeq(), FastSeq(Ref(Name("x"), TInt32)), Collect()),
          false,
        ),
        false,
      ),
      (agg, aggType),
      expected,
    )
  }

  test("keyedKeyedCallStats") {
    val agg = FastSeq(
      Row("EUR", "CASE", null),
      Row("EUR", "CONTROL", Call2(0, 1)),
      Row("AFR", "CASE", Call2(1, 1)),
      Row("AFR", "CONTROL", null),
    )
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "g" -> TCall)
    val expected = Map(
      "EUR" -> Map(
        "CONTROL" -> Row(FastSeq(1, 1), FastSeq(0.5, 0.5), 2, FastSeq(0, 0)),
        "CASE" -> Row(FastSeq(0, 0), null, 0, FastSeq(0, 0)),
      ),
      "AFR" -> Map(
        "CASE" -> Row(FastSeq(0, 2), FastSeq(0.0, 1.0), 2, FastSeq(0, 1)),
        "CONTROL" -> Row(FastSeq(0, 0), null, 0, FastSeq(0, 0)),
      ),
    )
    assertEvalsTo(
      AggGroupBy(
        Ref(Name("k1"), TString),
        AggGroupBy(
          Ref(Name("k2"), TString),
          ApplyAggOp(FastSeq(I32(2)), FastSeq(Ref(Name("g"), TCall)), CallStats()),
          false,
        ),
        false,
      ),
      (agg, aggType),
      expected,
    )
  }

  test("keyedKeyedTakeBy") {
    val agg = FastSeq(
      Row("case", "a", 0.2, 5),
      Row("control", "b", 0.4, 0),
      Row(null, "c", 1.0, 3),
      Row("control", "b", 0.0, 2),
      Row("case", "a", 0.3, 6),
      Row("control", "b", 0.5, 1),
    )
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "x" -> TFloat64, "y" -> TInt32)
    val expected = Map(
      "case" -> Map("a" -> FastSeq(0.2, 0.3)),
      "control" -> Map("b" -> FastSeq(0.4, 0.5)),
      (null, Map("c" -> FastSeq(1.0))),
    )
    assertEvalsTo(
      AggGroupBy(
        Ref(Name("k1"), TString),
        AggGroupBy(
          Ref(Name("k2"), TString),
          ApplyAggOp(
            FastSeq(I32(2)),
            FastSeq(Ref(Name("x"), TFloat64), Ref(Name("y"), TInt32)),
            TakeBy(),
          ),
          false,
        ),
        false,
      ),
      (agg, aggType),
      expected,
    )
  }

  test("keyedKeyedKeyedCollect") {
    val agg = FastSeq(
      Row("EUR", "CASE", true, 1),
      Row("EUR", "CONTROL", true, 2),
      Row("AFR", "CASE", false, 3),
      Row("AFR", "CONTROL", false, 4),
    )
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "k3" -> TBoolean, "x" -> TInt32)
    val expected = Map(
      "EUR" -> Map("CASE" -> Map(true -> FastSeq(1)), "CONTROL" -> Map(true -> FastSeq(2))),
      "AFR" -> Map("CASE" -> Map(false -> FastSeq(3)), "CONTROL" -> Map(false -> FastSeq(4))),
    )
    assertEvalsTo(
      AggGroupBy(
        Ref(Name("k1"), TString),
        AggGroupBy(
          Ref(Name("k2"), TString),
          AggGroupBy(
            Ref(Name("k3"), TBoolean),
            ApplyAggOp(FastSeq(), FastSeq(Ref(Name("x"), TInt32)), Collect()),
            false,
          ),
          false,
        ),
        false,
      ),
      (agg, aggType),
      expected,
    )
  }

  test("downsampleWhenEmpty") {
    runAggregator(
      Downsample(),
      TStruct("x" -> TFloat64, "y" -> TFloat64, "label" -> TArray(TString)),
      FastSeq(),
      FastSeq(),
      FastSeq(10),
      seqOpArgs = FastSeq(
        Ref(Name("x"), TFloat64),
        Ref(Name("y"), TFloat64),
        Ref(Name("label"), TArray(TString)),
      ),
    )
  }

  test("AggFilter") {
    val aggType = TStruct("x" -> TBoolean, "y" -> TInt64)
    val agg = FastSeq(Row(true, -1L), Row(true, 1L), Row(false, 3L), Row(true, 5L))

    assertEvalsTo(
      AggFilter(
        Ref(Name("x"), TBoolean),
        ApplyAggOp(FastSeq(), FastSeq(Ref(Name("y"), TInt64)), Sum()),
        false,
      ),
      (agg, aggType),
      5L,
    )
  }

  test("AggExplode") {
    val aggType = TStruct("x" -> TArray(TInt64))
    val agg = FastSeq(
      Row(FastSeq[Long](1, 4)),
      Row(FastSeq[Long]()),
      Row(FastSeq[Long](-1, 3)),
      Row(FastSeq[Long](4, 5, 6, -7)),
    )

    assertEvalsTo(
      aggExplodeIR(ToStream(Ref(Name("x"), TArray(TInt64))), false) { y =>
        ApplyAggOp(FastSeq(), FastSeq(y), Sum())
      },
      (agg, aggType),
      15L,
    )
  }

  test("ArrayElementsAggregator") {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int): IR = {
      val ht = TableRange(10, 3)
        .mapRows('row.insertFields('aRange -> irRange(0, m, 1)))

      TableAggregate(
        ht,
        aggArrayPerElement(
          GetField(Ref(TableIR.rowName, ht.typ.rowType), "aRange")
        )((elt, _) => ApplyAggOp(FastSeq(), FastSeq(Cast(elt, TInt64)), Sum())),
      )
    }

    assertEvalsTo(getAgg(10, 10), IndexedSeq.range(0, 10).map(_ * 10L))
  }

  test("ArrayElementsAggregatorEmpty") {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int, knownLength: Option[IR]): IR = {
      val ht = TableRange(10, 3)
        .mapRows('row.insertFields('aRange -> irRange(0, m, 1)))
        .mapGlobals('global.insertFields('m -> m))
        .filter(false)

      TableAggregate(
        ht,
        aggArrayPerElement(
          GetField(Ref(TableIR.rowName, ht.typ.rowType), "aRange"),
          knownLength,
        )((elt, _) => ApplyAggOp(FastSeq(), FastSeq(Cast(elt, TInt64)), Sum())),
      )
    }

    assertEvalsTo(getAgg(10, 10, None), null)
    assertEvalsTo(getAgg(10, 10, Some(1)), FastSeq(0L))
    assertEvalsTo(
      getAgg(10, 10, Some(GetField(Ref(TableIR.globalName, TStruct("m" -> TInt32)), "m"))),
      ArraySeq.fill(10)(0L),
    )
  }

  test("ImputeTypeSimple") {
    runAggregator(ImputeType(), TString, FastSeq(null), Row(false, false, true, true, true, true))
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("1231", "1234.5", null),
      Row(true, false, false, false, false, true),
    )
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("1231", "123"),
      Row(true, true, false, true, true, true),
    )
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("true", "false"),
      Row(true, true, true, false, false, false),
    )
  }

  test("FoldAgg") {
    val myIR = streamAggIR(
      mapIR(rangeIR(100))(idx => makestruct(("idx", idx), ("unused", idx + idx)))
    )(foo => aggFoldIR(I32(0))(_ + GetField(foo, "idx"))(_ + _))
    assertEvalsTo(myIR, 4950)

    val myTableIR = TableAggregate(
      TableRange(100, 5),
      aggFoldIR(I32(0)) { bar =>
        bar + GetField(Ref(TableIR.rowName, TStruct("idx" -> TInt32)), "idx")
      }(_ + _),
    )

    val analyses = LoweringAnalyses.apply(myTableIR, ctx)
    val myLoweredTableIR = LowerTableIR(myTableIR, DArrayLowering.All, ctx, analyses)

    assertEvalsTo(myLoweredTableIR, 4950)
  }

  test("FoldScan") {
    val foo = Ref(freshName(), TStruct("idx" -> TInt32, "unused" -> TInt32))

    val myIR = ToArray(
      StreamAggScan(
        mapIR(rangeIR(10))(idx => makestruct(("idx", idx), ("unused", idx + idx))),
        foo.name,
        aggFoldIR(I32(0), isScan = true)(bar => bar + GetField(foo, "idx"))(_ + _),
      )
    )
    assertEvalsTo(myIR, IndexedSeq(0, 0, 1, 3, 6, 10, 15, 21, 28, 36))
  }

  // fails because there is no "lowest binding referenced in an init op"
  test("StreamAgg") {
    implicit val execStrats = Set(ExecStrategy.JvmCompileUnoptimized)
    val foo = StreamRange(I32(0), I32(10), I32(1))

    val elt = Ref(freshName(), TInt32)
    val agg1 = bindIR(I32(1))(i => ApplyAggOp(Take(), i)(elt))
    val agg2 = bindIR(I32(2))(i => ApplyAggOp(Take(), i)(elt))
    val ir = StreamAgg(
      foo,
      elt.name,
      ApplyBinaryPrimOp(Add(), ArrayRef(agg1, I32(0)), ArrayRef(agg2, I32(1))),
    )
    assertEvalsTo(ir, 1)
  }

  test("LetBoundInitOpArg") {
    implicit val execStrats = ExecStrategy.allRelational
    var tir: TableIR = TableRange(10, 3)
    tir =
      TableMapRows(
        tir,
        InsertFields(Ref(TableIR.rowName, tir.typ.rowType), FastSeq("aStr" -> Str("foo"))),
      )
    tir = TableMapGlobals(tir, MakeStruct(FastSeq("n" -> I32(5))))
    val x = TableAggregate(
      tir,
      bindIR(GetField(Ref(TableIR.globalName, tir.typ.globalType), "n")) { n =>
        MakeTuple.ordered(FastSeq(
          n,
          ApplyAggOp(Take(), n)(GetField(Ref(TableIR.rowName, tir.typ.rowType), "idx")),
        ))
      },
    )

    assertEvalsTo(
      x,
      Row(
        5,
        0 until 5,
      ),
    )
  }
}

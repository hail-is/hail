package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.defs.{
  AggFilter, AggGroupBy, ApplyAggOp, ApplyBinaryPrimOp, ArrayRef, GetField, I32, InsertFields,
  MakeStruct, MakeTuple, Ref, Str, StreamAgg, StreamAggScan, StreamRange, TableAggregate, ToArray,
  ToStream,
}
import is.hail.expr.ir.lowering.{DArrayLowering, LowerTableIR}
import is.hail.types.virtual._
import is.hail.variant.Call2

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class AggregatorsSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.compileOnly

  def runAggregator(
    op: AggOp,
    aggType: TStruct,
    agg: IndexedSeq[Row],
    expected: Any,
    initOpArgs: IndexedSeq[IR],
    seqOpArgs: IndexedSeq[IR],
  )(implicit ctx: ExecuteContext
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
  )(implicit ctx: ExecuteContext
  ): Unit = {
    runAggregator(
      op,
      TStruct("x" -> t),
      a.map(i => RowSeq(i)),
      expected,
      initOpArgs,
      seqOpArgs = FastSeq(Ref(Name("x"), t)),
    )
  }

  @Test def nestedAgg(implicit ctx: ExecuteContext): Unit = {
    val agg = ToArray(mapIR(StreamRange(0, 10, 1))(_ => ApplyAggOp(Count())()))
    assertEvalsTo(
      agg,
      (FastSeq(1, 2).map(i => RowSeq(i)), TStruct("x" -> TInt32)),
      IndexedSeq.fill(10)(2L),
    )
  }

  @Test def sumFloat64(implicit ctx: ExecuteContext): Unit = {
    runAggregator(Sum(), TFloat64, (0 to 100).map(_.toDouble), 5050.0)
    runAggregator(Sum(), TFloat64, FastSeq(), 0.0)
    runAggregator(Sum(), TFloat64, FastSeq(42.0), 42.0)
    runAggregator(Sum(), TFloat64, FastSeq(null, 42.0, null), 42.0)
    runAggregator(Sum(), TFloat64, FastSeq(null, null, null), 0.0)
  }

  @Test def sumInt64(implicit ctx: ExecuteContext): Unit =
    runAggregator(Sum(), TInt64, FastSeq(-1L, 2L, 3L), 4L)

  @Test def collectBoolean(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Collect(),
      TBoolean,
      FastSeq(true, false, null, true, false),
      FastSeq(true, false, null, true, false),
    )
  }

  @Test def collectInt(implicit ctx: ExecuteContext): Unit =
    runAggregator(Collect(), TInt32, FastSeq(10, null, 5), FastSeq(10, null, 5))

  @Test def collectLong(implicit ctx: ExecuteContext): Unit =
    runAggregator(Collect(), TInt64, FastSeq(10L, null, 5L), FastSeq(10L, null, 5L))

  @Test def collectFloat(implicit ctx: ExecuteContext): Unit =
    runAggregator(Collect(), TFloat32, FastSeq(10f, null, 5f), FastSeq(10f, null, 5f))

  @Test def collectDouble(implicit ctx: ExecuteContext): Unit =
    runAggregator(Collect(), TFloat64, FastSeq(10d, null, 5d), FastSeq(10d, null, 5d))

  @Test def collectString(implicit ctx: ExecuteContext): Unit =
    runAggregator(Collect(), TString, FastSeq("hello", null, "foo"), FastSeq("hello", null, "foo"))

  @Test def collectArray(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Collect(),
      TArray(TInt32),
      FastSeq(FastSeq(1, 2, 3), null, FastSeq()),
      FastSeq(FastSeq(1, 2, 3), null, FastSeq()),
    )
  }

  @Test def collectStruct(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Collect(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastSeq(RowSeq(5, true), RowSeq(3, false), null, RowSeq(0, false), null),
      FastSeq(RowSeq(5, true), RowSeq(3, false), null, RowSeq(0, false), null),
    )
  }

  @Test def count(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Count(),
      TStruct("x" -> TString),
      FastSeq(
        RowSeq("hello"),
        RowSeq("foo"),
        RowSeq("a"),
        RowSeq(null),
        RowSeq("b"),
        RowSeq(null),
        RowSeq("c"),
      ),
      7L,
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )
  }

  @Test def collectAsSetBoolean(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      CollectAsSet(),
      TBoolean,
      FastSeq(true, false, null, true, false),
      Set(true, false, null),
    )
    runAggregator(CollectAsSet(), TBoolean, FastSeq(true, null, true), Set(true, null))
  }

  @Test def collectAsSetNumeric(implicit ctx: ExecuteContext): Unit = {
    runAggregator(CollectAsSet(), TInt32, FastSeq(10, null, 5, 5, null), Set(10, null, 5))
    runAggregator(CollectAsSet(), TInt64, FastSeq(10L, null, 5L, 5L, null), Set(10L, null, 5L))
    runAggregator(CollectAsSet(), TFloat32, FastSeq(10f, null, 5f, 5f, null), Set(10f, null, 5f))
    runAggregator(CollectAsSet(), TFloat64, FastSeq(10d, null, 5d, 5d, null), Set(10d, null, 5d))
  }

  @Test def collectAsSetString(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      CollectAsSet(),
      TString,
      FastSeq("hello", null, "foo", null, "foo"),
      Set("hello", null, "foo"),
    )
  }

  @Test def collectAsSetArray(implicit ctx: ExecuteContext): Unit = {
    val inputCollection = FastSeq(FastSeq(1, 2, 3), null, FastSeq(), null, FastSeq(1, 2, 3))
    val expected = Set(FastSeq(1, 2, 3), null, FastSeq())
    runAggregator(CollectAsSet(), TArray(TInt32), inputCollection, expected)
  }

  @Test def collectAsSetStruct(implicit ctx: ExecuteContext): Unit =
    runAggregator(
      CollectAsSet(),
      TStruct("a" -> TInt32, "b" -> TBoolean),
      FastSeq(RowSeq(5, true), RowSeq(3, false), null, RowSeq(0, false), null, RowSeq(5, true)),
      Set(RowSeq(5, true), RowSeq(3, false), null, RowSeq(0, false)),
    )

  @Test def callStats(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      CallStats(),
      TCall,
      FastSeq(Call2(0, 0), Call2(0, 1), null, Call2(0, 2)),
      RowSeq(FastSeq(4, 1, 1), FastSeq(4.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), 6, FastSeq(1, 0, 0)),
      initOpArgs = FastSeq(I32(3)),
    )
  }

  // FIXME Max Boolean not supported by old-style MaxAggregator

  @Test def maxInt32(implicit ctx: ExecuteContext): Unit = {
    runAggregator(Max(), TInt32, FastSeq(), null)
    runAggregator(Max(), TInt32, FastSeq(null), null)
    runAggregator(Max(), TInt32, FastSeq(-2, null, 7), 7)
  }

  @Test def maxInt64(implicit ctx: ExecuteContext): Unit =
    runAggregator(Max(), TInt64, FastSeq(-2L, null, 7L), 7L)

  @Test def maxFloat32(implicit ctx: ExecuteContext): Unit =
    runAggregator(Max(), TFloat32, FastSeq(-2.0f, null, 7.2f), 7.2f)

  @Test def maxFloat64(implicit ctx: ExecuteContext): Unit =
    runAggregator(Max(), TFloat64, FastSeq(-2.0, null, 7.2), 7.2)

  @Test def takeInt32(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TInt32,
      FastSeq(2, null, 7),
      FastSeq(2, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test def takeInt64(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TInt64,
      FastSeq(2L, null, 7L),
      FastSeq(2L, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test def takeFloat32(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TFloat32,
      FastSeq(2.0f, null, 7.2f),
      FastSeq(2.0f, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test def takeFloat64(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TFloat64,
      FastSeq(2.0, null, 7.2),
      FastSeq(2.0, null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test def takeCall(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TCall,
      FastSeq(Call2(0, 0), null, Call2(1, 0)),
      FastSeq(Call2(0, 0), null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test def takeString(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      Take(),
      TString,
      FastSeq("a", null, "b"),
      FastSeq("a", null),
      initOpArgs = FastSeq(I32(2)),
    )
  }

  @Test
  def sumMultivar(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(
      ApplyAggOp(
        FastSeq(),
        FastSeq(ApplyBinaryPrimOp(Multiply(), Ref(Name("a"), TFloat64), Ref(Name("b"), TFloat64))),
        Sum(),
      ),
      (
        FastSeq(RowSeq(1.0, 10.0), RowSeq(10.0, 10.0), RowSeq(null, 10.0)),
        TStruct("a" -> TFloat64, "b" -> TFloat64),
      ),
      110.0,
    )
  }

  private[this] def assertArraySumEvalsTo[T](
    eltType: Type,
    a: IndexedSeq[Seq[T]],
    expected: Seq[T],
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val aggregable = a.map(RowSeq(_))
    val structType = TStruct("foo" -> TArray(eltType))

    assertEvalsTo(
      aggArrayPerElement(Ref(Name("foo"), TArray(eltType))) { (elt, _) =>
        ApplyAggOp(FastSeq(), FastSeq(elt), Sum())
      },
      (aggregable, structType),
      expected,
    )
  }

  @Test
  def arraySumFloat64OnEmpty(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(),
      null,
    )

  @Test
  def arraySumFloat64OnSingletonMissing(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(null),
      null,
    )

  @Test
  def arraySumFloat64OnAllMissing(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Double](
      TFloat64,
      FastSeq(null, null, null),
      null,
    )

  @Test
  def arraySumInt64OnEmpty(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(),
      null,
    )

  @Test
  def arraySumInt64OnSingletonMissing(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(null),
      null,
    )

  @Test
  def arraySumInt64OnAllMissing(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo[Long](
      TInt64,
      FastSeq(null, null, null),
      null,
    )

  @Test
  def arraySumFloat64OnSmallArray(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo(
      TFloat64,
      FastSeq(
        FastSeq(1.0, 2.0),
        FastSeq(10.0, 20.0),
        null,
      ),
      FastSeq(11.0, 22.0),
    )

  @Test
  def arraySumInt64OnSmallArray(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo(
      TInt64,
      FastSeq(
        FastSeq(1L, 2L),
        FastSeq(10L, 20L),
        null,
      ),
      FastSeq(11L, 22L),
    )

  @Test
  def arraySumInt64FirstElementMissing(implicit ctx: ExecuteContext): Unit =
    assertArraySumEvalsTo(
      TInt64,
      FastSeq(
        null,
        FastSeq(1L, 33L),
        FastSeq(42L, 3L),
      ),
      FastSeq(43L, 36L),
    )

  private[this] def assertTakeByEvalsTo(
    aggType: Type,
    keyType: Type,
    n: Int,
    a: IndexedSeq[Row],
    expected: IndexedSeq[Any],
  )(implicit ctx: ExecuteContext
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

  @Test def takeByNGreater(implicit ctx: ExecuteContext): Unit =
    assertTakeByEvalsTo(TInt32, TInt32, 5, FastSeq(RowSeq(3, 4)), FastSeq(3))

  @Test def takeByBooleanBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TBoolean,
      3,
      FastSeq(RowSeq(false, true), RowSeq(null, null), RowSeq(true, false)),
      FastSeq(true, false, null),
    )
  }

  @Test def takeByBooleanInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TInt32,
      3,
      FastSeq(
        RowSeq(false, 0),
        RowSeq(null, null),
        RowSeq(true, 1),
        RowSeq(false, 3),
        RowSeq(true, null),
        RowSeq(null, 2),
      ),
      FastSeq(false, true, null),
    )
  }

  @Test def takeByBooleanLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TInt64,
      3,
      FastSeq(
        RowSeq(false, 0L),
        RowSeq(null, null),
        RowSeq(true, 1L),
        RowSeq(false, 3L),
        RowSeq(true, null),
        RowSeq(null, 2L),
      ),
      FastSeq(false, true, null),
    )
  }

  @Test def takeByBooleanFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TFloat32,
      3,
      FastSeq(
        RowSeq(false, 0f),
        RowSeq(null, null),
        RowSeq(true, 1f),
        RowSeq(false, 3f),
        RowSeq(true, null),
        RowSeq(null, 2f),
      ),
      FastSeq(false, true, null),
    )
  }

  @Test def takeByBooleanDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TFloat64,
      3,
      FastSeq(
        RowSeq(false, 0d),
        RowSeq(null, null),
        RowSeq(true, 1d),
        RowSeq(false, 3d),
        RowSeq(true, null),
        RowSeq(null, 2d),
      ),
      FastSeq(false, true, null),
    )
  }

  @Test def takeByBooleanAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TBoolean,
      TString,
      3,
      FastSeq(
        RowSeq(false, "a"),
        RowSeq(null, null),
        RowSeq(true, "b"),
        RowSeq(false, "d"),
        RowSeq(true, null),
        RowSeq(null, "c"),
      ),
      FastSeq(false, true, null),
    )
  }

  @Test def takeByIntBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TBoolean,
      2,
      FastSeq(RowSeq(3, true), RowSeq(null, null), RowSeq(null, false)),
      FastSeq(null, 3),
    )
  }

  @Test def takeByIntInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TInt32,
      3,
      FastSeq(
        RowSeq(3, 4),
        RowSeq(null, null),
        RowSeq(null, 2),
        RowSeq(11, 0),
        RowSeq(45, 1),
        RowSeq(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  @Test def takeByIntLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TInt64,
      3,
      FastSeq(
        RowSeq(3, 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq(11, 0L),
        RowSeq(45, 1L),
        RowSeq(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  @Test def takeByIntFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TFloat32,
      3,
      FastSeq(
        RowSeq(3, 4f),
        RowSeq(null, null),
        RowSeq(null, 2f),
        RowSeq(11, 0f),
        RowSeq(45, 1f),
        RowSeq(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  @Test def takeByIntDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TFloat64,
      3,
      FastSeq(
        RowSeq(3, 4d),
        RowSeq(null, null),
        RowSeq(null, 2d),
        RowSeq(11, 0d),
        RowSeq(45, 1d),
        RowSeq(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  @Test def takeByIntAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt32,
      TString,
      3,
      FastSeq(
        RowSeq(3, "d"),
        RowSeq(null, null),
        RowSeq(null, "c"),
        RowSeq(11, "a"),
        RowSeq(45, "b"),
        RowSeq(3, null),
      ),
      FastSeq(11, 45, null),
    )
  }

  @Test def takeByLongBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TBoolean,
      2,
      FastSeq(RowSeq(3L, true), RowSeq(null, null), RowSeq(null, false)),
      FastSeq(null, 3L),
    )
  }

  @Test def takeByLongInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TInt32,
      3,
      FastSeq(
        RowSeq(3L, 4),
        RowSeq(null, null),
        RowSeq(null, 2),
        RowSeq(11L, 0),
        RowSeq(45L, 1),
        RowSeq(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  @Test def takeByLongLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TInt64,
      3,
      FastSeq(
        RowSeq(3L, 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq(11L, 0L),
        RowSeq(45L, 1L),
        RowSeq(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  @Test def takeByLongFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TFloat32,
      3,
      FastSeq(
        RowSeq(3L, 4f),
        RowSeq(null, null),
        RowSeq(null, 2f),
        RowSeq(11L, 0f),
        RowSeq(45L, 1f),
        RowSeq(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  @Test def takeByLongDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TFloat64,
      3,
      FastSeq(
        RowSeq(3L, 4d),
        RowSeq(null, null),
        RowSeq(null, 2d),
        RowSeq(11L, 0d),
        RowSeq(45L, 1d),
        RowSeq(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  @Test def takeByLongAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TInt64,
      TString,
      3,
      FastSeq(
        RowSeq(3L, "d"),
        RowSeq(null, null),
        RowSeq(null, "c"),
        RowSeq(11L, "a"),
        RowSeq(45L, "b"),
        RowSeq(3L, null),
      ),
      FastSeq(11L, 45L, null),
    )
  }

  @Test def takeByFloatBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TBoolean,
      2,
      FastSeq(RowSeq(3f, true), RowSeq(null, null), RowSeq(null, false)),
      FastSeq(null, 3f),
    )
  }

  @Test def takeByFloatInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TInt32,
      3,
      FastSeq(
        RowSeq(3f, 4),
        RowSeq(null, null),
        RowSeq(null, 2),
        RowSeq(11f, 0),
        RowSeq(45f, 1),
        RowSeq(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  @Test def takeByFloatLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TInt64,
      3,
      FastSeq(
        RowSeq(3f, 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq(11f, 0L),
        RowSeq(45f, 1L),
        RowSeq(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  @Test def takeByFloatFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TFloat32,
      3,
      FastSeq(
        RowSeq(3f, 4f),
        RowSeq(null, null),
        RowSeq(null, 2f),
        RowSeq(11f, 0f),
        RowSeq(45f, 1f),
        RowSeq(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  @Test def takeByFloatDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TFloat64,
      3,
      FastSeq(
        RowSeq(3f, 4d),
        RowSeq(null, null),
        RowSeq(null, 2d),
        RowSeq(11f, 0d),
        RowSeq(45f, 1d),
        RowSeq(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  @Test def takeByFloatAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat32,
      TString,
      3,
      FastSeq(
        RowSeq(3f, "d"),
        RowSeq(null, null),
        RowSeq(null, "c"),
        RowSeq(11f, "a"),
        RowSeq(45f, "b"),
        RowSeq(3f, null),
      ),
      FastSeq(11f, 45f, null),
    )
  }

  @Test def takeByDoubleBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TBoolean,
      2,
      FastSeq(RowSeq(3d, true), RowSeq(null, null), RowSeq(null, false)),
      FastSeq(null, 3d),
    )
  }

  @Test def takeByDoubleInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TInt32,
      3,
      FastSeq(
        RowSeq(3d, 4),
        RowSeq(null, null),
        RowSeq(null, 2),
        RowSeq(11d, 0),
        RowSeq(45d, 1),
        RowSeq(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  @Test def takeByDoubleLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TInt64,
      3,
      FastSeq(
        RowSeq(3d, 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq(11d, 0L),
        RowSeq(45d, 1L),
        RowSeq(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  @Test def takeByDoubleFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TFloat32,
      3,
      FastSeq(
        RowSeq(3d, 4f),
        RowSeq(null, null),
        RowSeq(null, 2f),
        RowSeq(11d, 0f),
        RowSeq(45d, 1f),
        RowSeq(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  @Test def takeByDoubleDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TFloat64,
      3,
      FastSeq(
        RowSeq(3d, 4d),
        RowSeq(null, null),
        RowSeq(null, 2d),
        RowSeq(11d, 0d),
        RowSeq(45d, 1d),
        RowSeq(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  @Test def takeByDoubleAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TFloat64,
      TString,
      3,
      FastSeq(
        RowSeq(3d, "d"),
        RowSeq(null, null),
        RowSeq(null, "c"),
        RowSeq(11d, "a"),
        RowSeq(45d, "b"),
        RowSeq(3d, null),
      ),
      FastSeq(11d, 45d, null),
    )
  }

  @Test def takeByAnnotationBoolean(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TBoolean,
      2,
      FastSeq(RowSeq("hello", true), RowSeq(null, null), RowSeq(null, false)),
      FastSeq(null, "hello"),
    )
  }

  @Test def takeByAnnotationInt(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TInt32,
      3,
      FastSeq(
        RowSeq("a", 4),
        RowSeq(null, null),
        RowSeq(null, 2),
        RowSeq("b", 0),
        RowSeq("c", 1),
        RowSeq("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  @Test def takeByAnnotationLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TInt64,
      3,
      FastSeq(
        RowSeq("a", 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq("b", 0L),
        RowSeq("c", 1L),
        RowSeq("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  @Test def takeByAnnotationFloat(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TFloat32,
      3,
      FastSeq(
        RowSeq("a", 4f),
        RowSeq(null, null),
        RowSeq(null, 2f),
        RowSeq("b", 0f),
        RowSeq("c", 1f),
        RowSeq("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  @Test def takeByAnnotationDouble(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TFloat64,
      3,
      FastSeq(
        RowSeq("a", 4d),
        RowSeq(null, null),
        RowSeq(null, 2d),
        RowSeq("b", 0d),
        RowSeq("c", 1d),
        RowSeq("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  @Test def takeByAnnotationAnnotation(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TString,
      TString,
      3,
      FastSeq(
        RowSeq("a", "d"),
        RowSeq(null, null),
        RowSeq(null, "c"),
        RowSeq("b", "a"),
        RowSeq("c", "b"),
        RowSeq("d", null),
      ),
      FastSeq("b", "c", null),
    )
  }

  @Test def takeByCallLong(implicit ctx: ExecuteContext): Unit = {
    assertTakeByEvalsTo(
      TCall,
      TInt64,
      3,
      FastSeq(
        RowSeq(Call2(0, 0), 4L),
        RowSeq(null, null),
        RowSeq(null, 2L),
        RowSeq(Call2(0, 1), 0L),
        RowSeq(Call2(1, 1), 1L),
        RowSeq(Call2(0, 2), null),
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
  )(implicit ctx: ExecuteContext
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

  @Test
  def keyedCount(implicit ctx: ExecuteContext): Unit = {
    runKeyedAggregator(
      Count(),
      Ref(Name("k"), TInt32),
      TStruct("k" -> TInt32),
      FastSeq(RowSeq(1), RowSeq(2), RowSeq(3), RowSeq(1), RowSeq(1), RowSeq(null), RowSeq(null)),
      Map(1 -> 3L, 2 -> 1L, 3 -> 1L, (null, 2L)),
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )

    runKeyedAggregator(
      Count(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean),
      FastSeq(
        RowSeq(true),
        RowSeq(true),
        RowSeq(true),
        RowSeq(false),
        RowSeq(false),
        RowSeq(null),
        RowSeq(null),
      ),
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
        RowSeq(RowSeq(true)),
        RowSeq(RowSeq(true)),
        RowSeq(RowSeq(true)),
        RowSeq(RowSeq(false)),
        RowSeq(RowSeq(false)),
        RowSeq(RowSeq(null)),
        RowSeq(RowSeq(null)),
      ),
      Map(RowSeq(true) -> 3L, RowSeq(false) -> 2L, (RowSeq(null), 2L)),
      initOpArgs = FastSeq(),
      seqOpArgs = FastSeq(),
    )
  }

  @Test
  def keyedCollect(implicit ctx: ExecuteContext): Unit = {
    runKeyedAggregator(
      Collect(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean, "v" -> TInt32),
      FastSeq(
        RowSeq(true, 5),
        RowSeq(true, 3),
        RowSeq(true, null),
        RowSeq(false, 0),
        RowSeq(false, null),
        RowSeq(null, null),
        RowSeq(null, 2),
      ),
      Map(true -> FastSeq(5, 3, null), false -> FastSeq(0, null), (null, FastSeq(null, 2))),
      FastSeq(),
      FastSeq(Ref(Name("v"), TInt32)),
    )
  }

  @Test
  def keyedCallStats(implicit ctx: ExecuteContext): Unit = {
    runKeyedAggregator(
      CallStats(),
      Ref(Name("k"), TBoolean),
      TStruct("k" -> TBoolean, "v" -> TCall),
      FastSeq(
        RowSeq(true, null),
        RowSeq(true, Call2(0, 1)),
        RowSeq(true, Call2(0, 1)),
        RowSeq(false, null),
        RowSeq(false, Call2(0, 0)),
        RowSeq(false, Call2(1, 1)),
      ),
      Map(
        true -> RowSeq(FastSeq(2, 2), FastSeq(0.5, 0.5), 4, FastSeq(0, 0)),
        false -> RowSeq(FastSeq(2, 2), FastSeq(0.5, 0.5), 4, FastSeq(1, 1)),
      ),
      FastSeq(I32(2)),
      FastSeq(Ref(Name("v"), TCall)),
    )
  }

  @Test
  def keyedTakeBy(implicit ctx: ExecuteContext): Unit = {
    runKeyedAggregator(
      TakeBy(),
      Ref(Name("k"), TString),
      TStruct("k" -> TString, "x" -> TFloat64, "y" -> TInt32),
      FastSeq(
        RowSeq("case", 0.2, 5),
        RowSeq("control", 0.4, 0),
        RowSeq(null, 1.0, 3),
        RowSeq("control", 0.0, 2),
        RowSeq("case", 0.3, 6),
        RowSeq("control", 0.5, 1),
      ),
      Map("case" -> FastSeq(0.2, 0.3), "control" -> FastSeq(0.4, 0.5), (null, FastSeq(1.0))),
      FastSeq(I32(2)),
      FastSeq(Ref(Name("x"), TFloat64), Ref(Name("y"), TInt32)),
    )
  }

  @Test
  def keyedKeyedCollect(implicit ctx: ExecuteContext): Unit = {
    val agg =
      FastSeq(
        RowSeq("EUR", true, 1),
        RowSeq("EUR", false, 2),
        RowSeq("AFR", true, 3),
        RowSeq("AFR", null, 4),
      )
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

  @Test
  def keyedKeyedCallStats(implicit ctx: ExecuteContext): Unit = {
    val agg = FastSeq(
      RowSeq("EUR", "CASE", null),
      RowSeq("EUR", "CONTROL", Call2(0, 1)),
      RowSeq("AFR", "CASE", Call2(1, 1)),
      RowSeq("AFR", "CONTROL", null),
    )
    val aggType = TStruct("k1" -> TString, "k2" -> TString, "g" -> TCall)
    val expected = Map(
      "EUR" -> Map(
        "CONTROL" -> RowSeq(FastSeq(1, 1), FastSeq(0.5, 0.5), 2, FastSeq(0, 0)),
        "CASE" -> RowSeq(FastSeq(0, 0), null, 0, FastSeq(0, 0)),
      ),
      "AFR" -> Map(
        "CASE" -> RowSeq(FastSeq(0, 2), FastSeq(0.0, 1.0), 2, FastSeq(0, 1)),
        "CONTROL" -> RowSeq(FastSeq(0, 0), null, 0, FastSeq(0, 0)),
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

  @Test
  def keyedKeyedTakeBy(implicit ctx: ExecuteContext): Unit = {
    val agg = FastSeq(
      RowSeq("case", "a", 0.2, 5),
      RowSeq("control", "b", 0.4, 0),
      RowSeq(null, "c", 1.0, 3),
      RowSeq("control", "b", 0.0, 2),
      RowSeq("case", "a", 0.3, 6),
      RowSeq("control", "b", 0.5, 1),
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

  @Test
  def keyedKeyedKeyedCollect(implicit ctx: ExecuteContext): Unit = {
    val agg = FastSeq(
      RowSeq("EUR", "CASE", true, 1),
      RowSeq("EUR", "CONTROL", true, 2),
      RowSeq("AFR", "CASE", false, 3),
      RowSeq("AFR", "CONTROL", false, 4),
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

  @Test def downsampleWhenEmpty(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testAggFilter(implicit ctx: ExecuteContext): Unit = {
    val aggType = TStruct("x" -> TBoolean, "y" -> TInt64)
    val agg = FastSeq(RowSeq(true, -1L), RowSeq(true, 1L), RowSeq(false, 3L), RowSeq(true, 5L))

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

  @Test def testAggExplode(implicit ctx: ExecuteContext): Unit = {
    val aggType = TStruct("x" -> TArray(TInt64))
    val agg = FastSeq(
      RowSeq(FastSeq[Long](1, 4)),
      RowSeq(FastSeq[Long]()),
      RowSeq(FastSeq[Long](-1, 3)),
      RowSeq(FastSeq[Long](4, 5, 6, -7)),
    )

    assertEvalsTo(
      aggExplodeIR(ToStream(Ref(Name("x"), TArray(TInt64))), false) { y =>
        ApplyAggOp(FastSeq(), FastSeq(y), Sum())
      },
      (agg, aggType),
      15L,
    )
  }

  @Test def testArrayElementsAggregator(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int): IR =
      TableRange(n, 3)
        .mapRows((_, row) => row.insert("aRange" -> rangeIR(m).toArray))
        .aggregate((_, row) =>
          row.get("aRange").aggElements()((elt, _) => ApplyAggOp(Sum())(elt.toL))
        )

    assertEvalsTo(getAgg(10, 10), IndexedSeq.range(0, 10).map(_ * 10L))
  }

  @Test def testArrayElementsAggregatorEmpty(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.interpretOnly

    def getAgg(n: Int, m: Int, knownLength: Option[IR]): IR =
      TableRange(n, 3)
        .mapRows((_, row) => row.insert("aRange" -> rangeIR(m).toArray))
        .mapGlobals(_.insert("m" -> I32(m)))
        .filter((_, _) => false)
        .aggregate { (_, row) =>
          row.get("aRange").aggElements(knownLength)((elt, _) => ApplyAggOp(Sum())(elt.toL))
        }

    assertEvalsTo(getAgg(10, 10, None), null)
    assertEvalsTo(getAgg(10, 10, Some(1)), FastSeq(0L))
    assertEvalsTo(
      getAgg(10, 10, Some(GetField(Ref(TableIR.globalName, TStruct("m" -> TInt32)), "m"))),
      ArraySeq.fill(10)(0L),
    )
  }

  @Test def testImputeTypeSimple(implicit ctx: ExecuteContext): Unit = {
    runAggregator(
      ImputeType(),
      TString,
      FastSeq(null),
      RowSeq(false, false, true, true, true, true),
    )
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("1231", "1234.5", null),
      RowSeq(true, false, false, false, false, true),
    )
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("1231", "123"),
      RowSeq(true, true, false, true, true, true),
    )
    runAggregator(
      ImputeType(),
      TString,
      FastSeq("true", "false"),
      RowSeq(true, true, true, false, false, false),
    )
  }

  @Test def testFoldAgg(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testFoldScan(implicit ctx: ExecuteContext): Unit = {
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
  @Test def testStreamAgg(implicit ctx: ExecuteContext): Unit = {
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

  @Test def testLetBoundInitOpArg(implicit ctx: ExecuteContext): Unit = {
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
      RowSeq(
        5,
        0 until 5,
      ),
    )
  }
}

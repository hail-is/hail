package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.{BroadcastRow, RowSeq}
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.implicits._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.{Memoized => M}
import is.hail.expr.ir.Scope.{AGG, SCAN}
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{
  AggFilter, AggGroupBy, Apply, ApplyAggOp, ApplyScanOp, ArrayMaximalIndependentSet, Binding, Block,
  GetField, I32, I64, InsertFields, Let, Literal, MakeArray, MakeTuple, MatrixAggregate,
  MatrixMultiWrite, MatrixWrite, RNGSplit, RNGSplitStatic, RNGStateLiteral, Ref, TableAggregate,
  TableCollect, ToArray, ToStream, True,
}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.junit.jupiter.api.Test

class MatrixIRSuite {

  implicit val execStrats: Set[ExecStrategy] =
    Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def testMatrixWriteRead(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val range = MatrixIR.range(ctx, 10, 10, Some(3))
    val withEntries = MatrixMapEntries(
      range,
      makestruct(
        "i" -> GetField(Ref(MatrixIR.rowName, range.typ.rowType), "row_idx"),
        "j" -> GetField(Ref(MatrixIR.colName, range.typ.colType), "col_idx"),
      ),
    )
    val original = MatrixMapGlobals(withEntries, makestruct("foo" -> I32(0)))
    val path = ctx.createTmpPath("test-range-read", "mt")

    val writer1 = MatrixNativeWriter(path, overwrite = true)
    val partType = TArray(TInterval(TStruct("row_idx" -> TInt32)))
    val parts = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(
      FastSeq(Interval(RowSeq(0), RowSeq(10), true, false)),
      partType,
    ))
    val writer2 = MatrixNativeWriter(
      path,
      overwrite = true,
      partitions = parts.toString,
      partitionsTypeStr = partType.parsableString(),
    )

    ArraySeq(writer1, writer2).foreach { writer =>
      assertEvalsTo(MatrixWrite(original, writer), ())

      val read =
        MatrixIR.read(fs, ctx.references, path, dropCols = false, dropRows = false, None)
      val droppedRows =
        MatrixIR.read(fs, ctx.references, path, dropCols = false, dropRows = true, None)

      val expectedCols = ArraySeq.tabulate(10)(i => RowSeq(i, RowSeq(0L, i.toLong)))
      val expectedRows = if (writer eq writer1) {
        val uids = for {
          (partSize, partIndex) <- partition(10, 3).zipWithIndex
          i <- 0 until partSize
        } yield RowSeq(partIndex.toLong, i.toLong)
        (0 until 10).lazyZip(uids).map { (i, uid) =>
          RowSeq(i, uid, expectedCols.map { case Row(j, _) => RowSeq(i, j) })
        }
      } else
        ArraySeq.tabulate(10)(i =>
          RowSeq(i, RowSeq(0L, i.toLong), expectedCols.map { case Row(j, _) => RowSeq(i, j) })
        )
      val expectedGlobals = RowSeq(0, expectedCols);
      {
        implicit val execStrats: Set[ExecStrategy] =
          Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
        assertEvalsTo(
          TableCollect(TableKeyBy(CastMatrixToTable(read, "entries", "cols"), FastSeq())),
          RowSeq(expectedRows, expectedGlobals),
        )
        assertEvalsTo(
          TableCollect(TableKeyBy(CastMatrixToTable(droppedRows, "entries", "cols"), FastSeq())),
          RowSeq(FastSeq(), expectedGlobals),
        )
      }
    }
  }

  def rangeMatrix(
    nRows: Int = 20,
    nCols: Int = 20,
    nPartitions: Option[Int] = Some(4),
    uids: Boolean = false,
  )(implicit ctx: ExecuteContext
  ): MatrixIR = {
    val reader = MatrixRangeReader(ctx, nRows, nCols, nPartitions)
    val requestedType = if (uids)
      reader.fullMatrixType
    else
      reader.fullMatrixTypeWithoutUIDs
    MatrixRead(requestedType, false, false, reader)
  }

  def getRows(mir: MatrixIR)(implicit ctx: ExecuteContext): Array[Row] =
    Interpret(MatrixRowsTable(mir), ctx).rdd.collect()

  def getCols(mir: MatrixIR)(implicit ctx: ExecuteContext): Array[Row] =
    Interpret(MatrixColsTable(mir), ctx).rdd.collect()

  // For `assertEvalsTo`, which runs each of `execStrats`. TableCollect needs an unkeyed table.
  def collectRows(mir: MatrixIR): IR =
    GetField(TableCollect(TableKeyBy(MatrixRowsTable(mir), FastSeq())), "rows")

  def collectCols(mir: MatrixIR): IR =
    GetField(TableCollect(TableKeyBy(MatrixColsTable(mir), FastSeq())), "rows")

  // A range matrix with global `two = 2`, for init args that read `global`.
  def rangeMatrixWithGlobal(nRows: Int, nCols: Int)(implicit ctx: ExecuteContext): MatrixIR =
    MatrixMapGlobals(rangeMatrix(nRows, nCols, Some(2)), makestruct("two" -> I32(2)))

  def globalTwo: IR = Ref(MatrixIR.globalName, TStruct("two" -> TInt32)).get("two")

  @Test def testScanCountBehavesLikeIndexOnRows(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldRow = Ref(MatrixIR.rowName, mt.typ.rowType)

    val newRow = InsertFields(oldRow, FastSeq("idx" -> IRScanCount))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx }, rows.toSeq)
  }

  @Test def testScanCollectBehavesLikeRangeOnRows(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldRow = Ref(MatrixIR.rowName, mt.typ.rowType)

    val newRow =
      InsertFields(oldRow.ir, FastSeq("range" -> IRScanCollect(GetField(oldRow, "row_idx"))))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[_]) =>
      range sameElements Array.range(0, row_idx)
    })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnRows(implicit ctx: ExecuteContext)
    : Unit = {
    val mt = rangeMatrix()
    val oldRow = Ref(MatrixIR.rowName, mt.typ.rowType)

    val newRow = InsertFields(
      oldRow.ir,
      FastSeq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldRow, "row_idx").toL)),
    )

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, n: Long, range: IndexedSeq[_]) =>
      (n == 20) && (range sameElements Array.range(0, row_idx))
    })
  }

  @Test def testScanCountBehavesLikeIndexOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldCol = Ref(MatrixIR.colName, mt.typ.colType)

    val newCol = InsertFields(oldCol, FastSeq("idx" -> IRScanCount))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx, idx) => col_idx == idx })
  }

  @Test def testScanCollectBehavesLikeRangeOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldCol = Ref(MatrixIR.colName, mt.typ.colType)

    val newCol =
      InsertFields(oldCol.ir, FastSeq("range" -> IRScanCollect(GetField(oldCol, "col_idx"))))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, range: IndexedSeq[_]) =>
      range sameElements Array.range(0, col_idx)
    })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnCols(implicit ctx: ExecuteContext)
    : Unit = {
    val mt = rangeMatrix()
    val oldCol = Ref(MatrixIR.colName, mt.typ.colType)

    val newCol = InsertFields(
      oldCol.ir,
      FastSeq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldCol, "col_idx").toL)),
    )

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, n: Long, range: IndexedSeq[_]) =>
      (n == 20) && (range sameElements Array.range(0, col_idx))
    })
  }

  @Test def testStreamAggStreamReadsParentAggOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldRow = Ref(MatrixIR.rowName, mt.typ.rowType)
    val oldCol = Ref(MatrixIR.colName, mt.typ.colType)

    val sum =
      ApplyAggOp(Collect())(GetField(oldRow, "row_idx"))
        .stream
        .streamAgg(y => ApplyAggOp(Sum())(y.toL))

    val cols = getCols(MatrixMapCols(mt, InsertFields(oldCol, FastSeq("sum" -> sum)), None))
    assert(cols.forall { case Row(_, sum: Long) => sum == 190 })
  }

  @Test def testStreamAggScanQueryScanNotLiftedOnRows(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix()
    val oldRow = Ref(MatrixIR.rowName, mt.typ.rowType)

    val counts =
      rangeIR(3)
        .streamAggScan(_ => ApplyScanOp(Count())())
        .toArray

    val rows = getRows(MatrixMapRows(mt, InsertFields(oldRow, FastSeq("counts" -> counts))))
    assert(rows.forall { case Row(_, counts: IndexedSeq[_]) => counts == IndexedSeq(0L, 1L, 2L) })
  }

  def scanSumAndMax(idx: IR): IR =
    M.eval {
      for {
        x <- M.lift[SCAN.type](for { x <- idx } yield x)
        sum <- ApplyScanOp(Sum())(x.toL)
        max <- ApplyScanOp(Max())(x.toL)
      } yield maketuple(sum, max)
    }

  def expectedSumAndMax(n: Int): IndexedSeq[Row] =
    (0 until n).map(i => RowSeq(i, RowSeq((0 until i).sum.toLong, if (i == 0) null else i - 1L)))

  @Test def testScanBlockBindingOnRows(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val v = scanSumAndMax(row.get("row_idx"))
    val rows = getRows(MatrixMapRows(mt, InsertFields(row.deepCopy, FastSeq("v" -> v))))
    assertEq(rows.toSeq, expectedSumAndMax(5))
  }

  @Test def testScanBlockBindingOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val v = scanSumAndMax(col.get("col_idx"))
    val cols = getCols(MatrixMapCols(mt, InsertFields(col.deepCopy, FastSeq("v" -> v)), None))
    assertEq(cols.toSeq, expectedSumAndMax(5))
  }

  /* Init args that read `n_cols`, `global` or EVAL bindings in newRow: directly, transitively,
   * inside a combinator body or an agg's seq arg, and after a SCAN binding in the same Block. The
   * last two cases have aggs and no scans. */
  def testInitArgBoundInNewRow() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    def colIdx = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    def two: IR = I32(3) - I32(1)
    def scanTake2(x: IR): IR = bindIR(two)(ApplyScanOp(Take(), _)(x))
    def prev(i: Int) = 0 until math.min(i, 2)
    def expected(f: Int => Any) = (0 until 5).map(i => RowSeq(i, f(i)))
    val x = Ref(freshName(), TInt32)
    val k = Ref(freshName(), TInt32)
    FastSeq(
      (
        M.eval(for {
          n <- Ref(Name("n_cols"), TInt32) - I32(1)
          t <- ApplyScanOp(Take(), n)(rowIdx)
        } yield maketuple(t, n)),
        expected(i => RowSeq(prev(i), 2)),
      ),
      (
        takeWithTransitiveInitArg(ApplyScanOp(Take(), _)(rowIdx)),
        expected(i => RowSeq(0 until math.min(i, 1), 2)),
      ),
      (ApplyScanOp(Take(), globalTwo)(rowIdx), expected(prev)),
      (bindIR(globalTwo)(ApplyScanOp(Take(), _)(rowIdx)), expected(prev)),
      (AggFilter(True(), scanTake2(rowIdx), isScan = true), expected(prev)),
      (
        AggGroupBy(I32(0), scanTake2(rowIdx), isScan = true),
        expected(i => if (i == 0) Map() else Map(0 -> prev(i))),
      ),
      (
        aggArrayPerElement(MakeArray(rowIdx), Some(I32(1)), isScan = true) { (elt, _) =>
          scanTake2(elt)
        },
        expected(i => FastSeq(prev(i))),
      ),
      (
        ApplyAggOp(Sum())(bindIR(two)(n => (ApplyScanOp(Take(), n)(rowIdx).len + n).toL)),
        expected(i => 3L * (prev(i).length + 2)),
      ),
      (
        Block(
          FastSeq(Binding(x.name, rowIdx, SCAN), Binding(k.name, two)),
          ApplyScanOp(Take(), k)(x),
        ),
        expected(prev),
      ),
      (
        takeWithTransitiveInitArg(ApplyAggOp(Take(), _)(colIdx)),
        expected(_ => RowSeq(FastSeq(0), 2)),
      ),
      (
        AggFilter(True(), takeWithTransitiveInitArg(ApplyAggOp(Take(), _)(colIdx)), false),
        expected(_ => RowSeq(FastSeq(0), 2)),
      ),
    )
  }

  @ParameterizedTest
  def testInitArgBoundInNewRow(v: IR, expected: IndexedSeq[Row])(implicit ctx: ExecuteContext)
    : Unit = {
    val mt = rangeMatrixWithGlobal(5, 3)
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    assertEvalsTo(collectRows(MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))), expected)
  }

  def testScanAggNestingOnRows() = {
    val n = Ref(freshName(), TInt32)
    val m = Ref(freshName(), TInt32)
    val col = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32))
    val row = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32))
    FastSeq(
      (ApplyScanOp(Sum())(ApplyAggOp(Count())()), "aggregations inside scans"),
      (
        M.eval {
          for {
            x <- M.lift[SCAN.type](for { x <- ApplyAggOp(Count())() } yield x)
            y <- ApplyScanOp(Sum())(x)
          } yield y
        },
        "aggregations inside scans",
      ),
      (
        Block(FastSeq(Binding(n.name, ApplyAggOp(Count())().toI)), ApplyScanOp(Take(), n)(I32(0))),
        "aggregations inside scans",
      ),
      (
        Let(
          FastSeq(n.name -> ApplyAggOp(Count())().toI, m.name -> (n - I32(1))),
          ApplyScanOp(Take(), m)(I32(0)),
        ),
        "aggregations inside scans",
      ),
      (
        ApplyAggOp(Sum())(ApplyScanOp(Take(), col.get("col_idx"))(I32(0)).len.toL),
        "scan init argument refers to sa",
      ),
      (
        ApplyAggOp(Sum())(ApplyScanOp(Take(), row.get("row_idx"))(I32(0)).len.toL),
        "scan init argument refers to va",
      ),
    )
  }

  @ParameterizedTest
  def testScanAggNestingOnRows(v: IR, error: String)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    assertFatal(collectRows(MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))), error)
  }

  def testAggOfScanOnRows() = {
    val x = Ref(freshName(), TInt64)
    FastSeq(
      ApplyAggOp(Sum())(ApplyScanOp(Count())()),
      Block(FastSeq(Binding(x.name, ApplyScanOp(Count())(), AGG)), ApplyAggOp(Sum())(x)),
    )
  }

  @ParameterizedTest
  def testAggOfScanOnRows(v: IR)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val rows = getRows(MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v))))
    assertEq(rows.toSeq, (0 until 5).map(i => RowSeq(i, 3L * i)))
  }

  def testScanOfAggOnCols() = {
    val x = Ref(freshName(), TInt64)
    FastSeq(
      ApplyScanOp(Sum())(ApplyAggOp(Count())()),
      Block(FastSeq(Binding(x.name, ApplyAggOp(Count())(), SCAN)), ApplyScanOp(Sum())(x)),
    )
  }

  @ParameterizedTest
  def testScanOfAggOnCols(v: IR)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertFatal(collectCols(mapped), "aggregations inside scans")
  }

  def testAggOfScanOnCols() = {
    val x = Ref(freshName(), TInt64)
    FastSeq(
      ApplyAggOp(Sum())(ApplyScanOp(Count())()),
      Block(FastSeq(Binding(x.name, ApplyScanOp(Count())(), AGG)), ApplyAggOp(Sum())(x)),
    )
  }

  @ParameterizedTest
  def testAggOfScanOnCols(v: IR)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertFatal(collectCols(mapped), "scans inside aggregations")
  }

  // Children that drop eval drop the agg and scan scopes too.
  private def scansUnderDropEval = {
    val edges = Literal(TArray(TTuple(TInt32, TInt32)), FastSeq(Row(0, 1)))
    val tieBreaker = ApplyScanOp(Count())().toD
    FastSeq(
      aggFoldIR(I64(0))(_ + I64(1))((a, b) => a + b + ApplyScanOp(Count())()),
      ArrayMaximalIndependentSet(edges, Some((freshName(), freshName(), tieBreaker))),
    )
  }

  def testScanUnderDropEvalOnRows() = scansUnderDropEval

  @ParameterizedTest
  def testScanUnderDropEvalOnRows(v: IR)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val mapped = MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))
    assertThrows[AssertionError](collectRows(mapped), "ApplyScanOp outside a scan scope")
  }

  def testScanUnderDropEvalOnCols() = scansUnderDropEval

  @ParameterizedTest
  def testScanUnderDropEvalOnCols(v: IR)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertThrows[AssertionError](collectCols(mapped), "ApplyScanOp outside a scan scope")
  }

  // knownLength is an init arg, so it drops its own agg (scan) scope.
  def testSameKindInKnownLengthOnRows() = {
    val a = Literal(TArray(TInt32), FastSeq(0))
    FastSeq(
      (
        aggArrayPerElement(a, Some(ApplyAggOp(Count())().toI))((_, _) => ApplyAggOp(Count())()),
        "ApplyAggOp outside an agg scope",
      ),
      (
        aggArrayPerElement(a, Some(ApplyScanOp(Count())().toI), isScan = true) { (_, _) =>
          ApplyScanOp(Count())()
        },
        "ApplyScanOp outside a scan scope",
      ),
    )
  }

  @ParameterizedTest
  def testSameKindInKnownLengthOnRows(v: IR, error: String)(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val mapped = MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))
    assertThrows[AssertionError](collectRows(mapped), error)
  }

  def testInitArgReadsResultOnRows() =
    initArgsReadingResult(isScan = false)(
      Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    ) ++ initArgsReadingResult(isScan = true)(
      Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    )

  @ParameterizedTest
  def testInitArgReadsResultOnRows(v: IR)(implicit ctx: ExecuteContext): Unit = {
    // Without the optimiser, which can forward a binding into an init arg and change the error.
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val mapped = MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))
    assertFatal(collectRows(mapped), "depends on an? (aggregation|scan) result")
  }

  def testInitArgReadsResultOnCols() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    def colIdx = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    initArgsReadingResult(isScan = false)(rowIdx).map(_ -> "depends on an aggregation result") ++
      initArgsReadingResult(isScan = true)(colIdx).map(_ -> "depends on a scan result")
  }

  @ParameterizedTest
  def testInitArgReadsResultOnCols(v: IR, error: String)(implicit ctx: ExecuteContext): Unit = {
    // Without the optimiser, which can forward a binding into an init arg and change the error.
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertFatal(collectCols(mapped), error)
  }

  /* Init args that read `global` or EVAL bindings in newCol: directly, transitively, one binding
   * `n` copied into the agg init bindings and moved into the scan init bindings, inside a
   * combinator body, and after a SCAN (AGG) binding in the same Block. */
  def testInitArgBoundInNewCol() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    def colIdx = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    def two: IR = I32(3) - I32(1)
    def take2(take: IR => IR): IR = bindIR(two)(take(_))
    def prev(i: Int) = 0 until math.min(i, 2)
    def expected(f: Int => Any) = (0 until 5).map(i => RowSeq(i, f(i)))
    val x = Ref(freshName(), TInt32)
    val y = Ref(freshName(), TInt32)
    val k = Ref(freshName(), TInt32)
    val j = Ref(freshName(), TInt32)
    FastSeq(
      (
        M.eval(for {
          n <- I32(3) - I32(1)
          t <- ApplyScanOp(Take(), n)(colIdx)
        } yield maketuple(t, n)),
        expected(i => RowSeq(0 until math.min(i, 2), 2)),
      ),
      (
        M.eval(for {
          n <- I32(3) - I32(1)
          t <- ApplyAggOp(Take(), n)(rowIdx)
        } yield maketuple(t, n)),
        expected(_ => RowSeq(0 until 2, 2)),
      ),
      (
        takeWithTransitiveInitArg(ApplyScanOp(Take(), _)(colIdx)),
        expected(i => RowSeq(0 until math.min(i, 1), 2)),
      ),
      (
        takeWithTransitiveInitArg(ApplyAggOp(Take(), _)(rowIdx)),
        expected(_ => RowSeq(FastSeq(0), 2)),
      ),
      (
        M.eval(for {
          n <- I32(3) - I32(1)
          a <- ApplyAggOp(Take(), n)(rowIdx)
          s <- ApplyScanOp(Take(), n)(colIdx)
        } yield maketuple(a, s, n)),
        expected(i => RowSeq(0 until 2, 0 until math.min(i, 2), 2)),
      ),
      (
        maketuple(ApplyAggOp(Take(), globalTwo)(rowIdx), ApplyScanOp(Take(), globalTwo)(colIdx)),
        expected(i => RowSeq(0 until 2, prev(i))),
      ),
      (
        bindIR(globalTwo) { n =>
          maketuple(ApplyAggOp(Take(), n)(rowIdx), ApplyScanOp(Take(), n)(colIdx))
        },
        expected(i => RowSeq(0 until 2, prev(i))),
      ),
      (AggFilter(True(), take2(ApplyScanOp(Take(), _)(colIdx)), isScan = true), expected(prev)),
      (
        AggGroupBy(I32(0), take2(ApplyAggOp(Take(), _)(rowIdx)), isScan = false),
        expected(_ => Map(0 -> (0 until 2))),
      ),
      (
        aggArrayPerElement(MakeArray(rowIdx), Some(I32(1))) { (elt, _) =>
          take2(ApplyAggOp(Take(), _)(elt))
        },
        expected(_ => FastSeq(0 until 2)),
      ),
      (
        Block(
          FastSeq(Binding(x.name, colIdx, SCAN), Binding(k.name, two)),
          ApplyScanOp(Take(), k)(x),
        ),
        expected(prev),
      ),
      (
        Block(
          FastSeq(Binding(y.name, rowIdx, AGG), Binding(j.name, two)),
          ApplyAggOp(Take(), j)(y),
        ),
        expected(_ => 0 until 2),
      ),
    )
  }

  @ParameterizedTest
  def testInitArgBoundInNewCol(v: IR, expected: IndexedSeq[Row])(implicit ctx: ExecuteContext)
    : Unit = {
    /* With the optimiser on, `Interpret` walks some of the lowered IR itself, and it has no case
     * for StreamAgg or StreamAggScan. */
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrixWithGlobal(3, 5)
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertEvalsTo(collectCols(mapped), expected)
  }

  /* IR that runs its own aggregation: TableAggregate, MatrixAggregate (lowered to TableAggregate
   * before lifting), StreamAgg and StreamAggScan. Lifting leaves their queries alone and lifts a
   * scan in the stream argument. Each case sits beside an outer `agg.count()`, which is 3. */
  private def nestedAggContexts(idx: IR)(implicit ctx: ExecuteContext) = {
    def outer(x: IR): IR = maketuple(x, ApplyAggOp(Count())())
    def prevs: IR = ToStream(ApplyScanOp(Collect())(idx))
    FastSeq[(IR, Int => Any)](
      (outer(TableAggregate(TableRange(4, 1), ApplyAggOp(Count())())), _ => RowSeq(4L, 3L)),
      (
        outer(MatrixAggregate(rangeMatrix(2, 3, Some(1)), ApplyAggOp(Count())())),
        _ => RowSeq(6L, 3L),
      ),
      (ToArray(streamAggScanIR(prevs)(_ => ApplyScanOp(Count())())), i => 0L until i.toLong),
      (streamAggIR(prevs)(_ => ApplyAggOp(Count())()), _.toLong),
    )
  }

  def testNestedAggContextOnRows(implicit ctx: ExecuteContext) =
    nestedAggContexts(Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx"))

  @ParameterizedTest
  def testNestedAggContextOnRows(v: IR, expected: Int => Any)(implicit ctx: ExecuteContext)
    : Unit = {
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    assertEvalsTo(
      collectRows(MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))),
      (0 until 5).map(i => RowSeq(i, expected(i))),
    )
  }

  def testNestedAggContextOnCols(implicit ctx: ExecuteContext) =
    nestedAggContexts(Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx"))

  @ParameterizedTest
  def testNestedAggContextOnCols(v: IR, expected: Int => Any)(implicit ctx: ExecuteContext)
    : Unit = {
    /* With the optimiser on, `Interpret` walks some of the lowered IR itself, and it has no case
     * for StreamAgg or StreamAggScan. */
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    assertEvalsTo(
      collectCols(MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)),
      (0 until 5).map(i => RowSeq(i, expected(i))),
    )
  }

  @Test def testAggSeqArgReadsColOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val v = ApplyAggOp(Sum())(col.get("col_idx").toL)
    val cols = getCols(MatrixMapCols(mt, InsertFields(col.deepCopy, FastSeq("v" -> v)), None))
    assertEq(cols.toSeq, (0 until 5).map(i => RowSeq(i, 3L * i)))
  }

  @Test def testAggBlockBindingOnCols(implicit ctx: ExecuteContext): Unit = {
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val v = M.eval {
      for {
        x <- M.lift[AGG.type](for { x <- row.get("row_idx") } yield x)
        sum <- ApplyAggOp(Sum())(x.toL)
        count <- ApplyScanOp(Count())()
      } yield maketuple(sum, count)
    }
    val cols = getCols(MatrixMapCols(mt, InsertFields(col.deepCopy, FastSeq("v" -> v)), None))
    assertEq(cols.toSeq, (0 until 5).map(i => RowSeq(i, RowSeq(3L, i.toLong))))
  }

  def testInitArgReadsNRowsOnCols() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    def colIdx = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    def nRows = Ref(Name("n_rows"), TInt64)
    def viaEval(op: IR => IR): IR = M.eval(for {
      n <- nRows.toI
      t <- op(n)
    } yield t)
    val ops = FastSeq[IR => IR](
      n => ApplyAggOp(Take(), n)(rowIdx),
      n => ApplyScanOp(Take(), n)(colIdx),
    )
    ops.flatMap(op => FastSeq(op(nRows.toI), viaEval(op)))
  }

  @ParameterizedTest
  def testInitArgReadsNRowsOnCols(v: IR)(implicit ctx: ExecuteContext): Unit = {
    // Without the optimiser, which can forward a binding into an init arg and change the error.
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertFatal(collectCols(mapped), "init argument refers to n_rows")
  }

  // Init args that read the row or a loop variable, directly or through an EVAL binding.
  def testInitArgReadsVaryingOnRows() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    FastSeq(
      (ApplyScanOp(Take(), rowIdx)(rowIdx), "scan init argument refers to va"),
      (
        ToArray(mapIR(rangeIR(3))(i => ApplyScanOp(Take(), i)(rowIdx))),
        "scan init argument refers to .*, which can't leave newRow",
      ),
      (
        ToArray(mapIR(rangeIR(3))(i => bindIR(i + I32(1))(k => ApplyScanOp(Take(), k)(rowIdx)))),
        "scan init argument refers to .*, which can't leave newRow",
      ),
    )
  }

  @ParameterizedTest
  def testInitArgReadsVaryingOnRows(v: IR, error: String)(implicit ctx: ExecuteContext): Unit = {
    // Without the optimiser, which can forward a binding into an init arg and change the error.
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(5, 3, Some(2))
    val row = Ref(MatrixIR.rowName, mt.typ.rowType)
    val mapped = MatrixMapRows(mt, InsertFields(row, FastSeq("v" -> v)))
    assertFatal(collectRows(mapped), error)
  }

  // Init args that read the col or a loop variable.
  def testInitArgReadsVaryingOnCols() = {
    def rowIdx = Ref(MatrixIR.rowName, TStruct("row_idx" -> TInt32)).get("row_idx")
    def colIdx = Ref(MatrixIR.colName, TStruct("col_idx" -> TInt32)).get("col_idx")
    FastSeq(
      (ApplyAggOp(Take(), colIdx)(rowIdx), "init argument refers to sa"),
      (ApplyScanOp(Take(), colIdx)(colIdx), "init argument refers to sa"),
      (
        ToArray(mapIR(rangeIR(3))(i => ApplyAggOp(Take(), i)(rowIdx))),
        "init argument refers to .*, which can't leave newCol",
      ),
    )
  }

  @ParameterizedTest
  def testInitArgReadsVaryingOnCols(v: IR, error: String)(implicit ctx: ExecuteContext): Unit = {
    // Without the optimiser, which can forward a binding into an init arg and change the error.
    implicit val execStrats: Set[ExecStrategy] = ExecStrategy.unoptimizedRelational
    val mt = rangeMatrix(3, 5, Some(2))
    val col = Ref(MatrixIR.colName, mt.typ.colType)
    val mapped = MatrixMapCols(mt, InsertFields(col, FastSeq("v" -> v)), None)
    assertFatal(collectCols(mapped), error)
  }

  def rangeRowMatrix(start: Int, end: Int)(implicit ctx: ExecuteContext): MatrixIR = {
    val i = end - start
    val baseRange = rangeMatrix(i, 5, Some(math.max(1, math.min(4, i))))
    val row = Ref(MatrixIR.rowName, baseRange.typ.rowType)
    MatrixKeyRowsBy(
      MatrixMapRows(
        MatrixKeyRowsBy(baseRange, FastSeq()),
        InsertFields(
          row,
          FastSeq("row_idx" -> (GetField(row.ir, "row_idx") + start)),
        ),
      ),
      FastSeq("row_idx"),
    )
  }

  def testMatrixUnionRows() = ArraySeq(
    FastSeq(0 -> 0, 5 -> 7),
    FastSeq(0 -> 1, 5 -> 7),
    FastSeq(0 -> 6, 5 -> 7),
    FastSeq(2 -> 3, 0 -> 1, 5 -> 7),
    FastSeq(2 -> 4, 0 -> 3, 5 -> 7),
    FastSeq(3 -> 6, 0 -> 1, 5 -> 7),
  )

  @ParameterizedTest
  def testMatrixUnionRows(ranges: IndexedSeq[(Int, Int)])(implicit ctx: ExecuteContext): Unit = {
    val expectedOrdering = ranges.flatMap { case (start, end) =>
      Array.range(start, end)
    }.sorted

    val unioned = MatrixUnionRows(ranges.map { case (start, end) =>
      rangeRowMatrix(start, end)
    })
    val actualOrdering = getRows(unioned).map { case Row(i: Int) => i }

    assert(actualOrdering sameElements expectedOrdering)
  }

  def testMatrixExplode() = ArraySeq[(IndexedSeq[String], IndexedSeq[Integer])](
    (FastSeq("empty"), FastSeq()),
    (FastSeq("null"), null),
    (FastSeq("set"), FastSeq(1, 3)),
    (FastSeq("one"), FastSeq(3)),
    (FastSeq("na"), FastSeq(null)),
    (FastSeq("x", "y"), FastSeq(3)),
    (FastSeq("foo", "bar"), FastSeq(1, 3)),
    (FastSeq("a", "b", "c"), FastSeq()),
  )

  @ParameterizedTest("testMatrixExplode")
  def testMatrixExplodeRows(
    path: IndexedSeq[String],
    collection: IndexedSeq[Integer],
  )(implicit
    ctx: ExecuteContext
  ): Unit = {
    val range = rangeMatrix(5, 2, None)

    val field = path.init.foldRight(path.last -> toIRArray(collection))(_ -> IRStruct(_))
    val annotated =
      MatrixMapRows(range, InsertFields(Ref(MatrixIR.rowName, range.typ.rowType), FastSeq(field)))

    val q = annotated.typ.rowType.query(path: _*)
    val exploded =
      getRows(MatrixExplodeRows(annotated, path)).map(q(_).asInstanceOf[Integer])

    val expected = if (collection == null) Array[Integer]() else Array.fill(5)(collection).flatten
    assert(exploded sameElements expected)
  }

  @ParameterizedTest("testMatrixExplode")
  def testMatrixExplodeCols(
    path: IndexedSeq[String],
    collection: IndexedSeq[Integer],
  )(implicit
    ctx: ExecuteContext
  ): Unit = {
    var mt = rangeMatrix(5, 2, None)
    val field = path.init.foldRight(path.last -> toIRArray(collection))(_ -> IRStruct(_))
    mt = MatrixMapCols(mt, Ref(MatrixIR.colName, mt.typ.colType).insert(field), None)
    val q = mt.typ.colType.query(path: _*)
    val exploded = getCols(MatrixExplodeCols(mt, path)).map(q(_).asInstanceOf[Integer])
    val expected = if (collection == null) Array[Integer]() else Array.fill(2)(collection).flatten
    assert(exploded sameElements expected)
  }

  // these two items are helper for UnlocalizedEntries testing,
  def makeLocalizedTable(
    rdata: IndexedSeq[Row],
    cdata: IndexedSeq[Row],
  )(implicit
    ctx: ExecuteContext
  ): TableIR = {
    val sc = ctx.backend.asSpark.sc
    val rowRdd = sc.parallelize(rdata)
    val rowSig = TStruct(
      "row_idx" -> TInt32,
      "animal" -> TString,
      "__entries" -> TArray(TStruct("ent1" -> TString, "ent2" -> TFloat64)),
    )
    val keyNames = FastSeq("row_idx")

    val colSig = TStruct("col_idx" -> TInt32, "tag" -> TString)
    val globalType = TStruct(("__cols", TArray(colSig)))
    var tv = TableValue(ctx, rowSig, keyNames, rowRdd)
    tv = tv.copy(
      typ = tv.typ.copy(globalType = globalType),
      globals = BroadcastRow(ctx, RowSeq(cdata), globalType),
    )
    TableLiteral(tv, ctx.theHailClassLoader)
  }

  @Test def testCastTableToMatrix(implicit ctx: ExecuteContext): Unit = {
    val rdata = ArraySeq(
      RowSeq(1, "fish", FastSeq(RowSeq("a", 1.0), RowSeq("x", 2.0))),
      RowSeq(2, "cat", FastSeq(RowSeq("b", 0.0), RowSeq("y", 0.1))),
      RowSeq(3, "dog", FastSeq(RowSeq("c", -1.0), RowSeq("z", 30.0))),
    )
    val cdata = ArraySeq(
      RowSeq(1, "atag"),
      RowSeq(2, "btag"),
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab, "__entries", "__cols", ArraySeq("col_idx"))
    // cols are same
    val mtCols = Interpret(MatrixColsTable(mir), ctx).rdd.collect()
    assert(mtCols sameElements cdata)

    // Rows are same
    val mtRows = Interpret(MatrixRowsTable(mir), ctx).rdd.collect()
    assert(mtRows sameElements rdata.map(row => RowSeq.fromSeq(row.toSeq.take(2))))

    // Round trip
    val roundTrip = Interpret(CastMatrixToTable(mir, "__entries", "__cols"), ctx)
    val localRows = roundTrip.rdd.collect()
    assert(localRows sameElements rdata)
    val localCols = roundTrip.globals.javaValue.getAs[IndexedSeq[Row]](0)
    assert(localCols sameElements cdata)
  }

  @Test def testCastTableToMatrixErrors(implicit ctx: ExecuteContext): Unit = {
    val rdata = ArraySeq(
      RowSeq(1, "fish", FastSeq(RowSeq("x", 2.0))),
      RowSeq(2, "cat", FastSeq(RowSeq("b", 0.0), RowSeq("y", 0.1))),
      RowSeq(3, "dog", FastSeq(RowSeq("c", -1.0), RowSeq("z", 30.0))),
    )
    val cdata = ArraySeq(
      RowSeq(1, "atag"),
      RowSeq(2, "btag"),
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab, "__entries", "__cols", ArraySeq("col_idx"))

    // All rows must have the same number of elements in the entry field as colTab has rows
    interceptSpark("length mismatch between entry array and column array") {
      Interpret(mir, ctx).rvd.count()
    }

    // The entry field must be an array
    interceptFatal("") {
      TypeCheck(ctx, CastTableToMatrix(rowTab, "animal", "__cols", ArraySeq("col_idx")))
    }

    val rdata2 = ArraySeq(
      RowSeq(1, "fish", null),
      RowSeq(2, "cat", FastSeq(RowSeq("b", 0.0), RowSeq("y", 0.1))),
      RowSeq(3, "dog", FastSeq(RowSeq("c", -1.0), RowSeq("z", 30.0))),
    )
    val rowTab2 = makeLocalizedTable(rdata2, cdata)
    val mir2 = CastTableToMatrix(rowTab2, "__entries", "__cols", ArraySeq("col_idx"))

    interceptSpark("missing")(Interpret(mir2, ctx).rvd.count())
  }

  @Test def testMatrixFiltersWorkWithRandomness(implicit ctx: ExecuteContext): Unit = {
    val range = rangeMatrix(20, 20, Some(4), uids = true)
    def rand(rng: IR): IR =
      Apply("rand_bool", FastSeq.empty, FastSeq(RNGSplitStatic(rng, 0), 0.5), TBoolean)

    val colUID = GetField(Ref(MatrixIR.colName, range.typ.colType), MatrixReader.colUIDFieldName)
    val colRNG = RNGSplit(RNGStateLiteral(), colUID)
    val cols = Interpret(MatrixFilterCols(range, rand(colRNG)), ctx).toMatrixValue(
      range.typ.colKey
    ).nCols
    val rowUID = GetField(Ref(MatrixIR.rowName, range.typ.rowType), MatrixReader.rowUIDFieldName)
    val rowRNG = RNGSplit(RNGStateLiteral(), rowUID)
    val rows = Interpret(MatrixFilterRows(range, rand(rowRNG)), ctx).rvd.count()
    val entryRNG = RNGSplit(RNGStateLiteral(), MakeTuple.ordered(FastSeq(rowUID, colUID)))
    val entries = Interpret(
      MatrixEntriesTable(MatrixFilterEntries(range, rand(entryRNG))),
      ctx,
    ).rvd.count()

    assert(cols < 20 && cols > 0)
    assert(rows < 20 && rows > 0)
    assert(entries < 400 && entries > 0)
  }

  @Test def testMatrixRepartition(implicit ctx: ExecuteContext): Unit = {
    val range = rangeMatrix(11, 3, Some(10))

    val params = Array(
      1 -> RepartitionStrategy.SHUFFLE,
      1 -> RepartitionStrategy.COALESCE,
      5 -> RepartitionStrategy.SHUFFLE,
      5 -> RepartitionStrategy.NAIVE_COALESCE,
      10 -> RepartitionStrategy.SHUFFLE,
      10 -> RepartitionStrategy.COALESCE,
    )

    params.foreach { case (n, strat) =>
      unoptimized { ctx =>
        val rvd = Interpret(MatrixRepartition(range, n, strat), ctx).rvd
        assert(rvd.getNumPartitions == n, n -> strat)
        val values = rvd.collect(ctx).map(r => r.getAs[Int](0))
        assert(values.isSorted && values.length == 11, n -> strat)
      }
    }
  }

  /** Writes `nTargets` column slices of an `nRows` x `nCols` range matrix table and checks each one
    * against the slice of columns and entries it should have received.
    */
  def assertPartitionedColumnsRoundTrip(
    nRows: Int,
    nCols: Int,
    nParts: Int,
    nTargets: Int,
    expectedSizes: IndexedSeq[Int],
  )(implicit ctx: ExecuteContext
  ): Unit = {
    val fs = ctx.fs
    val range = MatrixIR.range(ctx, nRows, nCols, Some(nParts))
    val withEntries = MatrixMapEntries(
      range,
      makestruct(
        "i" -> GetField(Ref(MatrixIR.rowName, range.typ.rowType), "row_idx"),
        "j" -> GetField(Ref(MatrixIR.colName, range.typ.colType), "col_idx"),
      ),
    )
    val original = MatrixMapGlobals(withEntries, makestruct("foo" -> I32(0)))
    val path = ctx.createTmpPath("test-partitioned-columns", "mt")

    assertEvalsTo(
      MatrixWrite(original, MatrixNativePartitionedColumnsWriter(path, nTargets, overwrite = true)),
      (),
    )

    assert(expectedSizes.sum == nCols)
    assert(fs.listDirectory(path).length == nTargets)

    val uids = for {
      (partSize, partIndex) <- partition(nRows, nParts).zipWithIndex
      i <- 0 until partSize
    } yield RowSeq(partIndex.toLong, i.toLong)
    val bounds = expectedSizes.scanLeft(0)(_ + _);

    {
      implicit val execStrats: Set[ExecStrategy] =
        Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)

      MatrixNativePartitionedColumnsWriter.targetPaths(path, nTargets).zipWithIndex.foreach {
        case (target, t) =>
          val colIdxs = bounds(t) until bounds(t + 1)

          /* Column UIDs are synthesized by the reader from each table's own column positions, so
           * they restart at zero in every output; `col_idx` carries the original identity. */
          val cols = colIdxs.map(j => RowSeq(j, RowSeq(0L, (j - bounds(t)).toLong)))
          val expectedRows = (0 until nRows).lazyZip(uids).map { (i, uid) =>
            RowSeq(i, uid, colIdxs.map(j => RowSeq(i, j)))
          }
          val read = MatrixIR.read(
            fs,
            ctx.references,
            target,
            dropCols = false,
            dropRows = false,
            None,
          )
          assertEvalsTo(
            TableCollect(TableKeyBy(CastMatrixToTable(read, "entries", "cols"), FastSeq())),
            RowSeq(expectedRows, RowSeq(0, cols)),
          )

          /* Reading through an interval seeks via this output's own index and its `entries_offset`
           * annotation, which the sequential scan above never touches. Requesting the type without
           * UIDs keeps the expected rows independent of how the reader numbers skipped rows. */
          val skipped = 1
          val opts = NativeReaderOptions(
            FastSeq(Interval(RowSeq(skipped), RowSeq(nRows), true, false)),
            TInt32,
          )
          val reader = MatrixNativeReader(fs, ctx.references, target, Some(opts))
          val indexed = MatrixRead(reader.fullMatrixTypeWithoutUIDs, false, false, reader)
          assertEvalsTo(
            TableCollect(TableKeyBy(CastMatrixToTable(indexed, "entries", "cols"), FastSeq())),
            RowSeq(
              (skipped until nRows).map(i => RowSeq(i, colIdxs.map(j => RowSeq(i, j)))),
              RowSeq(0, colIdxs.map(j => RowSeq(j))),
            ),
          )
      }
    }
  }

  @Test def testMatrixPartitionedColumnsWriteReadUneven(implicit ctx: ExecuteContext): Unit =
    // 7 columns into 3 matrix tables: the first gets the remainder.
    assertPartitionedColumnsRoundTrip(10, 7, 3, 3, FastSeq(3, 2, 2))

  @Test def testMatrixPartitionedColumnsWriteReadEven(implicit ctx: ExecuteContext): Unit =
    assertPartitionedColumnsRoundTrip(10, 6, 3, 3, FastSeq(2, 2, 2))

  @Test def testMatrixPartitionedColumnsWriteReadOneColumnEach(implicit ctx: ExecuteContext): Unit =
    assertPartitionedColumnsRoundTrip(4, 2, 1, 2, FastSeq(1, 1))

  @Test def testMatrixPartitionedColumnsTooFewColumns(implicit ctx: ExecuteContext): Unit = {
    val range = MatrixIR.range(ctx, 5, 2, Some(2))
    val path = ctx.createTmpPath("test-partitioned-columns-too-few", "mt")
    interceptException[Throwable]("cannot split fewer than 3 columns") {
      assertEvalsTo(
        MatrixWrite(range, MatrixNativePartitionedColumnsWriter(path, 3, overwrite = true)),
        (),
      )
    }
  }

  @Test def testMatrixPartitionedColumnsOverwrite(implicit ctx: ExecuteContext): Unit = {
    val fs = ctx.fs
    val range = MatrixIR.range(ctx, 4, 8, Some(2))
    val path = ctx.createTmpPath("test-partitioned-columns-overwrite", "mt")

    assertEvalsTo(
      MatrixWrite(range, MatrixNativePartitionedColumnsWriter(path, 4, overwrite = true)),
      (),
    )
    assert(fs.listDirectory(path).length == 4)

    // Overwriting clears the container, so the two stale targets don't survive.
    assertEvalsTo(
      MatrixWrite(range, MatrixNativePartitionedColumnsWriter(path, 2, overwrite = true)),
      (),
    )
    assert(fs.listDirectory(path).length == 2)

    interceptException[Throwable]("already exists") {
      assertEvalsTo(MatrixWrite(range, MatrixNativePartitionedColumnsWriter(path, 2)), ())
    }
  }

  @Test def testMatrixMultiWriteDifferentTypesRaisesError(implicit ctx: ExecuteContext): Unit = {
    val vcf = importVCF(getTestResource("sample.vcf"))
    val range = rangeMatrix(10, 2, None)
    val path1 = ctx.createTmpPath("test1")
    val path2 = ctx.createTmpPath("test2")
    interceptException[Throwable]("of the same type") {
      TypeCheck(
        ctx,
        MatrixMultiWrite(FastSeq(vcf, range), MatrixNativeMultiWriter(IndexedSeq(path1, path2))),
      )
    }
  }
}

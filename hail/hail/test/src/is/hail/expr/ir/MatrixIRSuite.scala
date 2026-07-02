package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.{BroadcastRow, RowSeq}
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.collection.implicits._
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{
  Apply, GetField, I32, InsertFields, MakeTuple, MatrixMultiWrite, MatrixWrite, RNGSplit,
  RNGSplitStatic, RNGStateLiteral, Ref, TableCollect,
}
import is.hail.types.virtual._
import is.hail.utils._

import scala.collection.compat._

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

      val read = MatrixIR.read(fs, path, dropCols = false, dropRows = false, None)
      val droppedRows = MatrixIR.read(fs, path, dropCols = false, dropRows = true, None)

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

  @Test def testMatrixMultiWriteDifferentTypesRaisesError(implicit ctx: ExecuteContext): Unit = {
    val vcf = importVCF(getTestResource("sample.vcf"))
    val range = rangeMatrix(10, 2, None)
    val path1 = ctx.createTmpPath("test1")
    val path2 = ctx.createTmpPath("test2")
    intercept[HailException] {
      TypeCheck(
        ctx,
        MatrixMultiWrite(FastSeq(vcf, range), MatrixNativeMultiWriter(IndexedSeq(path1, path2))),
      )
    }: Unit
  }
}

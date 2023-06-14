package is.hail.expr.ir

import cats.syntax.all._
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.BroadcastRow
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.lowering.Lower.monadLowerInstanceForLower
import is.hail.expr.ir.lowering.LoweringState
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.{ExecStrategy, HailSuite}
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods
import org.testng.annotations.{DataProvider, Test}

class MatrixIRSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def testMatrixWriteRead(): Unit = {
    val range = MatrixIR.range(10, 10, Some(3))
    val withEntries = MatrixMapEntries(range, makestruct(
      "i" -> GetField(Ref("va", range.typ.rowType), "row_idx"),
      "j" -> GetField(Ref("sa", range.typ.colType), "col_idx")))
    val original = MatrixMapGlobals(withEntries, makestruct("foo" -> I32(0)))
    val path = ctx.createTmpPath("test-range-read", "mt")

    val writer1 = MatrixNativeWriter(path, overwrite = true)
    val partType = TArray(TInterval(TStruct("row_idx" -> TInt32)))
    val parts = JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(FastIndexedSeq(Interval(Row(0), Row(10), true, false)), partType))
    val writer2 = MatrixNativeWriter(path, overwrite = true, partitions = parts.toString, partitionsTypeStr = partType.parsableString())

    for (writer <- Array(writer1, writer2)) {
      assertEvalsTo(MatrixWrite(original, writer), ())

      val read = MatrixIR.read(fs, path, dropCols = false, dropRows = false, None)
      val droppedRows = MatrixIR.read(fs, path, dropCols = false, dropRows = true, None)

      val expectedCols = Array.tabulate(10)(i => Row(i, Row(0L, i.toLong))).toFastIndexedSeq
      val expectedRows = if (writer eq writer1) {
        val uids = for {
          (partSize, partIndex) <- partition(10, 3).zipWithIndex
          i <- 0 until partSize
        } yield Row(partIndex.toLong, i.toLong)
        (0 until 10, uids).zipped.map { (i, uid) => Row(i, uid, expectedCols.map { case Row(j, _) => Row(i, j) }) }
      } else
        Array.tabulate(10)(i => Row(i, Row(0L, i.toLong), expectedCols.map { case Row(j, _) => Row(i, j) })).toFastIndexedSeq
      val expectedGlobals = Row(0, expectedCols);
      {
        implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized)
        assertEvalsTo(
          TableCollect(TableKeyBy(CastMatrixToTable(read, "entries", "cols"), FastIndexedSeq())),
          Row(expectedRows, expectedGlobals))
        assertEvalsTo(TableCollect(TableKeyBy(CastMatrixToTable(droppedRows, "entries", "cols"), FastIndexedSeq())), Row(FastIndexedSeq(), expectedGlobals))
      }
    }
  }

  def rangeMatrix(nRows: Int = 20, nCols: Int = 20, nPartitions: Option[Int] = Some(4), uids: Boolean = false): MatrixIR = {
    val reader = MatrixRangeReader(nRows, nCols, nPartitions)
    val requestedType = if (uids)
      reader.fullMatrixType
    else
      reader.fullMatrixTypeWithoutUIDs
    MatrixRead(requestedType, false, false, reader)
  }

  def getRows(mir: MatrixIR): Array[Row] =
    Interpret(MatrixRowsTable(mir)).runA(ctx, LoweringState()).rdd.collect()

  def getCols(mir: MatrixIR): Array[Row] =
    Interpret(MatrixColsTable(mir)).runA(ctx, LoweringState()).rdd.collect()

  @Test def testScanCountBehavesLikeIndexOnRows() {
    val mt = rangeMatrix()
    val oldRow = Ref("va", mt.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx }, rows.toSeq)
  }

  @Test def testScanCollectBehavesLikeRangeOnRows() {
    val mt = rangeMatrix()
    val oldRow = Ref("va", mt.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "row_idx"))))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[_]) => range sameElements Array.range(0, row_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnRows() {
    val mt = rangeMatrix()
    val oldRow = Ref("va", mt.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldRow, "row_idx").toL)))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, n: Long, range: IndexedSeq[_]) => (n == 20) && (range sameElements Array.range(0, row_idx)) })
  }

  @Test def testScanCountBehavesLikeIndexOnCols() {
    val mt = rangeMatrix()
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx, idx) => col_idx == idx })
  }

  @Test def testScanCollectBehavesLikeRangeOnCols() {
    val mt = rangeMatrix()
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("range" -> IRScanCollect(GetField(oldCol, "col_idx"))))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, range: IndexedSeq[_]) => range sameElements Array.range(0, col_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnCols() {
    val mt = rangeMatrix()
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldCol, "col_idx").toL)))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, n: Long, range: IndexedSeq[_]) => (n == 20) && (range sameElements Array.range(0, col_idx)) })
  }

  def rangeRowMatrix(start: Int, end: Int): MatrixIR = {
    val i = end - start
    val baseRange = rangeMatrix(i, 5, Some(math.max(1, math.min(4, i))))
    val row = Ref("va", baseRange.typ.rowType)
    MatrixKeyRowsBy(
      MatrixMapRows(
        MatrixKeyRowsBy(baseRange, FastIndexedSeq()),
        InsertFields(
          row,
          FastIndexedSeq("row_idx" -> (GetField(row, "row_idx") + start)))),
      FastIndexedSeq("row_idx"))
  }

  @DataProvider(name = "unionRowsData")
  def unionRowsData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(0 -> 0, 5 -> 7)),
    Array(FastIndexedSeq(0 -> 1, 5 -> 7)),
    Array(FastIndexedSeq(0 -> 6, 5 -> 7)),
    Array(FastIndexedSeq(2 -> 3, 0 -> 1, 5 -> 7)),
    Array(FastIndexedSeq(2 -> 4, 0 -> 3, 5 -> 7)),
    Array(FastIndexedSeq(3 -> 6, 0 -> 1, 5 -> 7)))

  @Test(dataProvider = "unionRowsData")
  def testMatrixUnionRows(ranges: IndexedSeq[(Int, Int)]) {
    val expectedOrdering = ranges.flatMap { case (start, end) =>
      Array.range(start, end)
    }.sorted

    val unioned = MatrixUnionRows(ranges.map { case (start, end) =>
      rangeRowMatrix(start, end)
    })
    val actualOrdering = getRows(unioned).map { case Row(i: Int) => i }

    assert(actualOrdering sameElements expectedOrdering)
  }

  @DataProvider(name = "explodeRowsData")
  def explodeRowsData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq("empty"), FastIndexedSeq()),
    Array(FastIndexedSeq("null"), null),
    Array(FastIndexedSeq("set"), FastIndexedSeq(1, 3)),
    Array(FastIndexedSeq("one"), FastIndexedSeq(3)),
    Array(FastIndexedSeq("na"), FastIndexedSeq(null)),
    Array(FastIndexedSeq("x", "y"), FastIndexedSeq(3)),
    Array(FastIndexedSeq("foo", "bar"), FastIndexedSeq(1, 3)),
    Array(FastIndexedSeq("a", "b", "c"), FastIndexedSeq()))

  @Test(dataProvider = "explodeRowsData")
  def testMatrixExplode(path: IndexedSeq[String], collection: IndexedSeq[Integer]) {
    val tarray = TArray(TInt32)
    val range = rangeMatrix(5, 2, None)

    val field = path.init.foldRight(path.last -> toIRArray(collection))(_ -> IRStruct(_))
    val annotated = MatrixMapRows(range, InsertFields(Ref("va", range.typ.rowType), FastIndexedSeq(field)))

    val q = annotated.typ.rowType.query(path: _*)
    val exploded = getRows(MatrixExplodeRows(annotated, path.toFastIndexedSeq)).map(q(_).asInstanceOf[Integer])

    val expected = if (collection == null) Array[Integer]() else Array.fill(5)(collection).flatten
    assert(exploded sameElements expected)
  }

  // these two items are helper for UnlocalizedEntries testing,
  def makeLocalizedTable(rdata: Array[Row], cdata: Array[Row]): TableIR = {
    val rowRdd = sc.parallelize(rdata)
    val rowSig = TStruct(
      "row_idx" -> TInt32,
      "animal" -> TString,
      "__entries" -> TArray(TStruct("ent1" -> TString, "ent2" -> TFloat64))
    )
    val keyNames = FastIndexedSeq("row_idx")

    val colSig = TStruct("col_idx" -> TInt32, "tag" -> TString)
    val globalType = TStruct(("__cols", TArray(colSig)))
    (for {
      tv_ <- TableValue[Execute](rowSig, keyNames, rowRdd);
      br <- BroadcastRow[Execute](Row(cdata.toFastIndexedSeq), globalType)
      tv = tv_.copy(typ = tv_.typ.copy(globalType = globalType), globals = br)
      lit <- TableLiteral[Execute](tv)
    } yield lit).apply(ctx)
  }

  @Test def testCastTableToMatrix() {
    val rdata = Array(
      Row(1, "fish", FastIndexedSeq(Row("a", 1.0), Row("x", 2.0))),
      Row(2, "cat", FastIndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", FastIndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val cdata = Array(
      Row(1, "atag"),
      Row(2, "btag")
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab, "__entries", "__cols", Array("col_idx"))
    // cols are same
    val mtCols = Interpret(MatrixColsTable(mir)).runA(ctx, LoweringState()).rdd.collect()
    assert(mtCols sameElements cdata)

    // Rows are same
    val mtRows = Interpret(MatrixRowsTable(mir)).runA(ctx, LoweringState()).rdd.collect()
    assert(mtRows sameElements rdata.map(row => Row.fromSeq(row.toSeq.take(2))))

    // Round trip
    val roundTrip = Interpret(CastMatrixToTable(mir, "__entries", "__cols")).runA(ctx, LoweringState())
    val localRows = roundTrip.rdd.collect()
    assert(localRows sameElements rdata)
    val localCols = roundTrip.globals.javaValue.getAs[IndexedSeq[Row]](0)
    assert(localCols sameElements cdata)
  }

  @Test def testCastTableToMatrixErrors() {
    val rdata = Array(
      Row(1, "fish", FastIndexedSeq(Row("x", 2.0))),
      Row(2, "cat", FastIndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", FastIndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val cdata = Array(
      Row(1, "atag"),
      Row(2, "btag")
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab, "__entries", "__cols", Array("col_idx"))

    // All rows must have the same number of elements in the entry field as colTab has rows
    interceptSpark("length mismatch between entry array and column array") {
      Interpret(mir, optimize = true).runA(ctx, LoweringState()).rvd.count()
    }

    // The entry field must be an array
    interceptFatal("") {
      CastTableToMatrix(rowTab, "animal", "__cols", Array("col_idx"))
    }

    val rdata2 = Array(
      Row(1, "fish", null),
      Row(2, "cat", FastIndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", FastIndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val rowTab2 = makeLocalizedTable(rdata2, cdata)
    val mir2 = CastTableToMatrix(rowTab2, "__entries", "__cols", Array("col_idx"))

    interceptSpark("missing") { Interpret(mir2, optimize = true).runA(ctx, LoweringState()).rvd.count() }
  }

  @Test def testMatrixFiltersWorkWithRandomness() {
    val range = rangeMatrix(20, 20, Some(4), uids = true)
    def rand(rng: IR): IR =
      ApplySeeded("rand_bool", FastIndexedSeq(0.5), rng, 0, TBoolean)

    val colUID = GetField(Ref("sa", range.typ.colType), MatrixReader.colUIDFieldName)
    val colRNG = RNGSplit(RNGStateLiteral(), colUID)
    val cols = Interpret(MatrixFilterCols(range, rand(colRNG)), optimize = true)
      .runA(ctx, LoweringState())
      .toMatrixValue(range.typ.colKey).nCols
    val rowUID = GetField(Ref("va", range.typ.rowType), MatrixReader.rowUIDFieldName)
    val rowRNG = RNGSplit(RNGStateLiteral(), rowUID)
    val rows = Interpret(MatrixFilterRows(range, rand(rowRNG)), optimize = true)
      .runA(ctx, LoweringState())
      .rvd.count()
    val entryRNG = RNGSplit(RNGStateLiteral(), MakeTuple.ordered(FastSeq(rowUID, colUID)))
    val entries = Interpret(MatrixEntriesTable(MatrixFilterEntries(range, rand(entryRNG))), optimize = true)
      .runA(ctx, LoweringState()).rvd.count()

    assert(cols < 20 && cols > 0)
    assert(rows < 20 && rows > 0)
    assert(entries < 400 && entries > 0)
  }

  @Test def testMatrixRepartition() {
    val range = rangeMatrix(11, 3, Some(10))

    val params = Array(
      1 -> RepartitionStrategy.SHUFFLE,
      1 -> RepartitionStrategy.COALESCE,
      5 -> RepartitionStrategy.SHUFFLE,
      5 -> RepartitionStrategy.NAIVE_COALESCE,
      10 -> RepartitionStrategy.SHUFFLE,
      10 -> RepartitionStrategy.COALESCE
    )
    params.foreach { case (n, strat) =>
      val rvd = Interpret(MatrixRepartition(range, n, strat), optimize = false)
        .runA(ctx, LoweringState()).rvd
      assert(rvd.getNumPartitions == n, n -> strat)
      val values = rvd.collect(ctx).map(r => r.getAs[Int](0))
      assert(values.isSorted && values.length == 11, n -> strat)
    }
  }

  @Test def testMatrixMultiWriteDifferentTypesRaisesError() {
    val vcf = is.hail.TestUtils.importVCF(ctx, "src/test/resources/sample.vcf")
    val range = rangeMatrix(10, 2, None)
    val path1 = ctx.createTmpPath("test1")
    val path2 = ctx.createTmpPath("test2")
    intercept[java.lang.IllegalArgumentException] {
      val ir = MatrixMultiWrite(FastIndexedSeq(vcf, range), MatrixNativeMultiWriter(IndexedSeq(path1, path2)))
    }
  }
}

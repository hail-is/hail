package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.ir.TestUtils._
import is.hail.utils._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class MatrixIRSuite extends SparkSuite {

  def rangeMatrix: MatrixIR = MatrixIR.range(hc, 20, 20, Some(4))

  def getRows(mir: MatrixIR): Array[Row] =
    Interpret(MatrixRowsTable(mir)).rdd.collect()

  def getCols(mir: MatrixIR): Array[Row] =
    Interpret(MatrixColsTable(mir)).rdd.collect()

  @Test def testScanCountBehavesLikeIndexOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx })
  }

  @Test def testScanCollectBehavesLikeRangeOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "row_idx"))))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[_]) => range sameElements Array.range(0, row_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldRow, "row_idx").toL)))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, n: Long, range: IndexedSeq[_]) => (n == 20) && (range sameElements Array.range(0, row_idx)) })
  }

  @Test def testScanCountBehavesLikeIndexOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx, idx) => col_idx == idx })
  }

  @Test def testScanCollectBehavesLikeRangeOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("range" -> IRScanCollect(GetField(oldCol, "col_idx"))))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, range: IndexedSeq[_]) => range sameElements Array.range(0, col_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldCol, "col_idx").toL)))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, n: Long, range: IndexedSeq[_]) => (n == 20) && (range sameElements Array.range(0, col_idx)) })
  }

  def rangeRowMatrix(start: Int, end: Int): MatrixIR = {
    val i = end - start
    val baseRange = MatrixIR.range(hc, i, 5, Some(math.min(4, i)))
    val row = Ref("va", baseRange.typ.rvRowType)
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
    val tarray = TArray(TInt32())
    val range = MatrixIR.range(hc, 5, 2, None)

    val field = path.init.foldRight(path.last -> toIRArray(collection))(_ -> IRStruct(_))
    val annotated = MatrixMapRows(range, InsertFields(Ref("va", range.typ.rvRowType), FastIndexedSeq(field)))

    val q = annotated.typ.rowType.query(path: _*)
    val exploded = getRows(MatrixExplodeRows(annotated, path.toIndexedSeq)).map(q(_).asInstanceOf[Integer])

    val expected = if (collection == null) Array[Integer]() else Array.fill(5)(collection).flatten
    assert(exploded sameElements expected)
  }

  @Test def testMatrixFiltersWorkWithRandomness() {
    val range = MatrixIR.range(hc, 20, 20, Some(4))
    val rand = ApplySeeded("rand_bool", FastIndexedSeq(0.5), seed=0)

    val cols = Interpret(MatrixFilterCols(range, rand)).nCols
    val rows = Interpret(MatrixFilterRows(range, rand)).rvd.count()
    val entries = Interpret(MatrixEntriesTable(MatrixFilterEntries(range, rand))).rvd.count()

    assert(cols < 20 && cols > 0)
    assert(rows < 20 && rows > 0)
    assert(entries < 400 && entries > 0)
  }

  @Test def testMatrixAggregateColsByKeyWithEntriesPosition() {
    val range = MatrixIR.range(hc, 3, 3, Some(1))
    val withEntries = MatrixMapEntries(range, MakeStruct(FastIndexedSeq("x" -> 2)))
    val m = MatrixAggregateColsByKey(
      MatrixMapRows(withEntries,
        InsertFields(Ref("va", withEntries.typ.rvRowType), FastIndexedSeq("a" -> 1))),
      MakeStruct(FastIndexedSeq("foo" -> IRAggCount)),
      MakeStruct(FastIndexedSeq("bar" -> IRAggCount))
    )
    assert(Interpret(m).rowsRVD().count() == 3)
  }

  @Test def testMatrixRepartition() {
    val range = MatrixIR.range(hc, 11, 3, Some(10))

    val params = Array(
      1 -> RepartitionStrategy.SHUFFLE,
      1 -> RepartitionStrategy.COALESCE,
      5 -> RepartitionStrategy.SHUFFLE,
      5 -> RepartitionStrategy.NAIVE_COALESCE,
      10 -> RepartitionStrategy.SHUFFLE,
      10 -> RepartitionStrategy.COALESCE
    )
    params.foreach { case (n, strat) =>
      val rvd = Interpret(MatrixRepartition(range, n, strat), optimize = false).rvd
      assert(rvd.getNumPartitions == n, n -> strat)
      val values = rvd.collect(CodecSpec.default).map(r => r.getAs[Int](0))
      assert(values.isSorted && values.length == 11, n -> strat)
    }
  }

  @Test def testMatrixMultiWriteDifferentTypesFails() {
    val vcf = is.hail.TestUtils.importVCF(hc, "src/test/resources/sample.vcf")
    val range = MatrixIR.range(hc, 10, 2, None)
    val path = tmpDir.createLocalTempFile()
    intercept[java.lang.IllegalArgumentException] {
      val ir = MatrixMultiWrite(IndexedSeq(vcf, range), MatrixNativeMultiWriter(path))
    }
  }
}

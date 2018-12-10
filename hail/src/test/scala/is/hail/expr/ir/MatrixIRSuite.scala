package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.expr.types.virtual._
import is.hail.io.CodecSpec
import is.hail.variant.MatrixTable
import org.apache.spark.SparkException
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class MatrixIRSuite extends SparkSuite {

  def rangeMatrix: MatrixIR = MatrixTable.range(hc, 20, 20, Some(4)).ast

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
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, row_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldRow, "row_idx").toL)))

    val newMatrix = MatrixMapRows(mt, newRow)
    val rows = getRows(newMatrix)
    assert(rows.forall { case Row(row_idx: Int, n: Long, range: IndexedSeq[Int]) => (n == 20) && (range sameElements Array.range(0, row_idx)) })
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
    assert(cols.forall { case Row(col_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, col_idx) })
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldCol, "col_idx").toL)))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = getCols(newMatrix)
    assert(cols.forall { case Row(col_idx: Int, n: Long, range: IndexedSeq[Int]) => (n == 20) && (range sameElements Array.range(0, col_idx)) })
  }

  def rangeRowMatrix(start: Int, end: Int): MatrixIR = {
    val i = end - start
    val baseRange = MatrixTable.range(hc, i, 5, Some(math.min(4, i))).ast
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
    val range = MatrixTable.range(hc, 5, 2, None).ast

    val field = path.init.foldRight(path.last -> toIRArray(collection))(_ -> IRStruct(_))
    val annotated = MatrixMapRows(range, InsertFields(Ref("va", range.typ.rvRowType), FastIndexedSeq(field)))

    val q = annotated.typ.rowType.query(path: _*)
    val exploded = getRows(MatrixExplodeRows(annotated, path.toIndexedSeq)).map(q(_).asInstanceOf[Integer])

    val expected = if (collection == null) Array[Integer]() else Array.fill(5)(collection).flatten
    assert(exploded sameElements expected)
  }

  // these two items are helper for UnlocalizedEntries testing,
  def makeLocalizedTable(rdata: Array[Row], cdata: Array[Row]): Table = {
    val rowRdd = sc.parallelize(rdata)
    val rowSig = TStruct(
      "row_idx" -> TInt32(),
      "animal" -> TString(),
      "__entries" -> TArray(TStruct("ent1" -> TString(), "ent2" -> TFloat64()))
    )
    val keyNames = IndexedSeq("row_idx")

    val colSig = TStruct("col_idx" -> TInt32(), "tag" -> TString())

    Table(hc, rowRdd, rowSig, keyNames, TStruct(("__cols", TArray(colSig))), Row(cdata.toIndexedSeq))
  }

  @Test def testCastTableToMatrix() {
    val rdata = Array(
      Row(1, "fish", IndexedSeq(Row("a", 1.0), Row("x", 2.0))),
      Row(2, "cat", IndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", IndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val cdata = Array(
      Row(1, "atag"),
      Row(2, "btag")
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab.tir, "__entries", "__cols", Array("col_idx"))
    // cols are same
    val mtCols = Interpret(MatrixColsTable(mir)).rdd.collect()
    assert(mtCols sameElements cdata)

    // Rows are same
    val mtRows = Interpret(MatrixRowsTable(mir)).rdd.collect()
    assert(mtRows sameElements rdata.map(row => Row.fromSeq(row.toSeq.take(2))))

    // Round trip
    val roundTrip = Interpret(CastMatrixToTable(mir, "__entries", "__cols"))
    val localRows = roundTrip.rdd.collect()
    assert(localRows sameElements rdata)
    val localCols = roundTrip.globals.value.getAs[IndexedSeq[Row]](0)
    assert(localCols sameElements cdata)
  }

  @Test def testCastTableToMatrixErrors() {
    val rdata = Array(
      Row(1, "fish", IndexedSeq(Row("x", 2.0))),
      Row(2, "cat", IndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", IndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val cdata = Array(
      Row(1, "atag"),
      Row(2, "btag")
    )
    val rowTab = makeLocalizedTable(rdata, cdata)

    val mir = CastTableToMatrix(rowTab.tir, "__entries", "__cols", Array("col_idx"))

    // All rows must have the same number of elements in the entry field as colTab has rows
    interceptSpark("incorrect entry array length") {
      Interpret(mir).rvd.count()
    }

    // The entry field must be an array
    interceptFatal("") {
      CastTableToMatrix(rowTab.tir, "animal", "__cols", Array("col_idx"))
    }

    val rdata2 = Array(
      Row(1, "fish", null),
      Row(2, "cat", IndexedSeq(Row("b", 0.0), Row("y", 0.1))),
      Row(3, "dog", IndexedSeq(Row("c", -1.0), Row("z", 30.0)))
    )
    val rowTab2 = makeLocalizedTable(rdata2, cdata)
    val mir2 = CastTableToMatrix(rowTab2.tir, "__entries", "__cols", Array("col_idx"))

    interceptSpark("missing") { Interpret(mir2).rvd.count() }
  }

  @Test def testMatrixFiltersWorkWithRandomness() {
    val range = MatrixTable.range(hc, 20, 20, Some(4)).ast
    val rand = ApplySeeded("rand_bool", FastIndexedSeq(0.5), seed=0)

    val cols = Interpret(MatrixFilterCols(range, rand)).nCols
    val rows = Interpret(MatrixFilterRows(range, rand)).rvd.count()
    val entries = Interpret(MatrixEntriesTable(MatrixFilterEntries(range, rand))).rvd.count()

    assert(cols < 20 && cols > 0)
    assert(rows < 20 && rows > 0)
    assert(entries < 400 && entries > 0)
  }

  @Test def testMatrixAggregateColsByKeyWithEntriesPosition() {
    val range = MatrixTable.range(hc, 3, 3, Some(1)).ast
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
    val range = MatrixTable.range(hc, 11, 3, Some(10)).ast

    val params = Array(
      1 -> true,
      1 -> false,
      5 -> true,
      5 -> false,
      10 -> true,
      10 -> false
    )
    params.foreach { case (n, shuffle) =>
      val rvd = Interpret(MatrixRepartition(range, n, shuffle), optimize = false).rvd
      assert(rvd.getNumPartitions == n, n -> shuffle)
      val values = rvd.collect(CodecSpec.default).map(r => r.getAs[Int](0))
      assert(values.isSorted && values.length == 11, n -> shuffle)
    }
  }

  @Test def testMatrixNativeWrite() {
    val range = MatrixTable.range(hc, 11, 3, Some(10))
    val path = tmpDir.createLocalTempFile(extension = "mt")
    Interpret(MatrixWrite(range.ast, MatrixNativeWriter(path)))
    val read = MatrixTable.read(hc, path)
    assert(read.same(range))
  }

  @Test def testMatrixVCFWrite() {
    val vcf = is.hail.TestUtils.importVCF("src/test/resources/sample.vcf")
    val path = tmpDir.createLocalTempFile(extension = "vcf")
    Interpret(MatrixWrite(vcf.ast, MatrixVCFWriter(path)))
  }

  @Test def testMatrixPLINKWrite() {
    val plinkPath = "src/test/resources/skip_invalid_loci"
    val plink = hc.importPlink(plinkPath + ".bed", plinkPath + ".bim", plinkPath + ".fam", skipInvalidLoci = true)
    val plinkIR = MatrixMapRows(plink.ast, InsertFields(Ref("va", plink.rvRowType),
      FastIndexedSeq("varid" -> GetField(Ref("va", plink.rvRowType), "rsid"))))
    val path = tmpDir.createLocalTempFile()
    Interpret(MatrixWrite(plinkIR, MatrixPLINKWriter(path)))
  }

  @Test def testMatrixGENWrite() {
    val gen = hc.importGen("src/test/resources/example.gen", "src/test/resources/example.sample",
      contigRecoding = Some(Map("01" -> "1")))
    val genIR = MatrixMapCols(gen.ast, SelectFields(InsertFields(Ref("sa", gen.colType),
      FastIndexedSeq("id1" -> Str(""), "id2" -> Str(""), "missing" -> F64(0.0))), FastIndexedSeq("id1", "id2", "missing")),
      Some(FastIndexedSeq("id1")))
    val path = tmpDir.createLocalTempFile()
    Interpret(MatrixWrite(genIR, MatrixGENWriter(path)))
  }
}

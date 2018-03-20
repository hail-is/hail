package is.hail.methods

import is.hail.expr.types.{TInt32, TInt64, TString, TStruct}
import breeze.linalg.{DenseMatrix => BDM}
import is.hail.{SparkSuite, TestUtils}
import is.hail.linalg.BlockMatrix
import is.hail.table.Table
import is.hail.testUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class VariantIntervalsSuite extends SparkSuite {

  def makeTableWithIndexLocusSchema(data: IndexedSeq[(String, Int)]): Table = {
    val rows = data.map { case (contig, pos) => Row(contig, pos) }
    Table.parallelize(hc, rows, TStruct("contig" -> TString(), "pos" -> TInt32()), IndexedSeq[String](), None)
      .index().select(Array("row.index", "row.contig", "row.pos"))
  }

  def makeSquareBlockMatrix(nRows: Int, blockSize: Int): BlockMatrix = {
    val arbitraryEntries = new BDM[Double](nRows, nRows, Array.fill[Double](nRows * nRows)(0))
    BlockMatrix.fromBreezeMatrix(sc, arbitraryEntries, blockSize)
  }

  def removeRequiredness(t: Table): Table = {
    // FIXME: requiredness shouldn't be needed for join
    val reqRowType = t.signature.fields.map(f => (f.name, f.typ.setRequired(false)))
    t.copy(signature = TStruct(t.signature.required, reqRowType: _*))
  }

  def joinOn(t: Table, tField: String, u: Table, uField: String): Table = {
    t.keyBy(tField).join(u.rename(Map(uField -> tField), Map.empty[String, String]).keyBy(tField), "left")
  }

  def renamePrependString(str: String, t: Table, fieldsToRename: Array[String]): Table = {
    val map = fieldsToRename.map { field => field -> (str + field.substring(0, 1).toUpperCase + field.substring(1)) }.toMap
    t.rename(map, Map.empty[String, String])
  }

  @Test def testIndexedLocusNearby() {
    val indexedLocusA = new IndexedLocus(0, "X", 100)
    assert(indexedLocusA.near(new IndexedLocus(1, "X", 105), window = 5))
    assert(!indexedLocusA.near(new IndexedLocus(2, "Y", 99), window = 5))
    assert(!indexedLocusA.near(new IndexedLocus(3, "X", 106), window = 5))
  }

  @Test def testIntervalsExcludeLociWithDifferentChromosomes() {
    val rows = IndexedSeq[(String, Int)](("X", 5), ("X", 6), ("Y", 7))
    val tbl = makeTableWithIndexLocusSchema(rows)

    val intervals = new VariantIntervals().computeIntervalsOfNearbyLociByIndex(tbl, window = 1)
    assert(intervals.toSet == Set((0, 1), (2, 2)))
  }

  @Test def testIntervalsWithSameStartAreCoalesced() {
    val rows = IndexedSeq[(String, Int)](("X", 5), ("X", 7), ("X", 13), ("X", 14), ("X", 17),
      ("X", 65), ("X", 70), ("X", 73), ("Y", 74), ("Y", 75), ("Y", 200), ("Y", 300))

    val tbl = makeTableWithIndexLocusSchema(rows)
    val intervals = new VariantIntervals().computeIntervalsOfNearbyLociByIndex(tbl, window = 10)
    val expected = Set((0, 3), (1, 4), (5, 7), (8, 9), (10, 10), (11, 11))
    assert(intervals.toSet == expected)
  }

  @Test def testGetIntervalsOfNearbyLociInDataset() {
    val tbl = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/sample.vcf.bgz")).rowsTable().index()
      .annotate("contig = row.locus.contig, pos = row.locus.position")
      .select(Array("row.index", "row.contig", "row.pos"))

    val window = 100000
    val intervals = new VariantIntervals().computeIntervalsOfNearbyLociByIndex(tbl, window)

    val result = intervals.flatMap { case (start, end) =>
      Array.fill(2)(start to end).flatten.combinations(2).flatMap(_.permutations)
    }.map(arr => (arr(0), arr(1))).distinct.map { case (a, b) => Array(a, b) }.toArray

    val closeVariants = s"(row.iContig == row.jContig) && ((abs(row.iPos - row.jPos)) <= $window)"

    val manuallyFilteredTable = renamePrependString("i", tbl, Array("index", "contig", "pos"))
      .join(renamePrependString("j", tbl, Array("index", "contig", "pos")), "inner")
      .filter(closeVariants, keep = true)

    val expectedRows = manuallyFilteredTable.select(Array("row.iIndex", "row.jIndex")).collect()
      .map(r => Array(r.get(0).asInstanceOf[Long], r.get(1).asInstanceOf[Long]))

    assert(result.sortBy(f => (f(0), f(1))).deep == expectedRows.sortBy(f => (f(0), f(1))).deep)
  }

  @Test def testEntriesTableFilterByWindow() {
    val tbl = TestUtils.splitMultiHTS(hc.importVCF("src/test/resources/sample.vcf.bgz")).rowsTable().index()
      .annotate("contig = row.locus.contig, pos = row.locus.position")
      .select(Array("row.index", "row.contig", "row.pos"))

    val bm = makeSquareBlockMatrix(tbl.count().toInt, blockSize = 10)

    val window = 100000
    val resultTable = removeRequiredness(EntriesTableFilterByWindow.apply(hc, tbl, bm, window))

    val matchVariantsToIndices = (t: Table) => {
      joinOn(joinOn(t, "i", tbl, "index")
        .rename(Map("contig" -> "iContig", "pos" -> "iPos"), Map.empty[String, String]), "j", tbl, "index")
        .rename(Map("contig" -> "jContig", "pos" -> "jPos"), Map.empty[String, String])
    }

    val removeDuplicate = "row.iPos < row.jPos"
    val closeVariants = s"(row.iContig == row.jContig) && (abs(row.iPos - row.jPos) <= $window)"

    // resultTable contains entire partitions, even if we only need part of partition, so filter it
    val filteredResultTable = matchVariantsToIndices(resultTable)
      .filter(s"($closeVariants) && ($removeDuplicate)", keep = true)

    val fullTable = removeRequiredness(bm.entriesTable(hc))

    val expectedTable = matchVariantsToIndices(fullTable)
      .filter(s"($closeVariants) && ($removeDuplicate)", keep = true)

    assert(filteredResultTable.select(Array("row.iContig", "row.iPos", "row.jContig", "row.jPos")).collect().toSet
      == expectedTable.select(Array("row.iContig", "row.iPos", "row.jContig", "row.jPos")).collect().toSet)
  }
}

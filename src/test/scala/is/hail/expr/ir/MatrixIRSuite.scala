package is.hail.expr.ir

import is.hail.SparkSuite
import is.hail.expr.ir.TestUtils._
import is.hail.table.Table
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class MatrixIRSuite extends SparkSuite {

  def rangeMatrix: MatrixIR = MatrixTable.range(hc, 20, 20, Some(4)).ast

  @Test def testScanCountBehavesLikeIndexOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = MatrixRowsTable(newMatrix).execute(hc).rdd.collect()
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx})
  }

  @Test def testScanCollectBehavesLikeRangeOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("range" -> IRScanCollect(GetField(oldRow, "row_idx"))))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = MatrixRowsTable(newMatrix).execute(hc).rdd.collect()
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, row_idx)})
  }

  @Test def testScanCollectBehavesLikeRangeWithAggregationOnRows() {
    val mt = rangeMatrix
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldRow, "row_idx").toL)))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = MatrixRowsTable(newMatrix).execute(hc).rdd.collect()
    assert(rows.forall { case Row(row_idx: Int, n: Long, range: IndexedSeq[Int]) => (n == 20) && (range sameElements Array.range(0, row_idx))})
  }

  @Test def testScanCountBehavesLikeIndexOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("idx" -> IRScanCount))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = MatrixColsTable(newMatrix).execute(hc).rdd.collect()
    assert(cols.forall { case Row(col_idx, idx) => col_idx == idx})
  }

  @Test def testScanCollectBehavesLikeRangeOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("range" -> IRScanCollect(GetField(oldCol, "col_idx"))))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = MatrixColsTable(newMatrix).execute(hc).rdd.collect()
    assert(cols.forall { case Row(col_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, col_idx)})
  }


  @Test def testScanCollectBehavesLikeRangeWithAggregationOnCols() {
    val mt = rangeMatrix
    val oldCol = Ref("sa", mt.typ.colType)

    val newCol = InsertFields(oldCol, Seq("n" -> IRAggCount, "range" -> IRScanCollect(GetField(oldCol, "col_idx").toL)))

    val newMatrix = MatrixMapCols(mt, newCol, None)
    val cols = MatrixColsTable(newMatrix).execute(hc).rdd.collect()
    assert(cols.forall { case Row(col_idx: Int, n: Long, range: IndexedSeq[Int]) => (n == 20) && (range sameElements Array.range(0, col_idx))})
  }
}

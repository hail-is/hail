package is.hail.expr

import is.hail.SparkSuite
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class MatrixIRSuite extends SparkSuite {

  def getMatrixTable: MatrixIR = MatrixTable.range(hc, 20, 20, Some(4)).ast

  def irAggCount: IR = {
    val aggSig = AggSignature(Count(), TInt32(), FastSeq.empty, None, FastSeq.empty)
    ApplyAggOp(SeqOp(0, 0, aggSig), FastIndexedSeq.empty, None, aggSig)
  }

  def irScanCount: IR = {
    val aggSig = AggSignature(Count(), TInt32(), FastSeq.empty, None, FastSeq.empty)
    ApplyScanOp(SeqOp(0, 0, aggSig), FastIndexedSeq.empty, None, aggSig)
  }

  def irAggCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), ir.typ, FastIndexedSeq(), None, FastIndexedSeq())
    ApplyAggOp(
      SeqOp(ir, I32(0), aggSig, FastIndexedSeq()),
      FastIndexedSeq(), None, aggSig)
  }

  def irScanCollect(ir: IR): IR = {
    val aggSig = AggSignature(Collect(), ir.typ, FastIndexedSeq(), None, FastIndexedSeq())
    ApplyScanOp(
      SeqOp(ir, I32(0), aggSig, FastIndexedSeq()),
      FastIndexedSeq(), None, aggSig)
  }

  @Test def testScanCountBehavesLikeIndex() {
    val mt = getMatrixTable
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("idx" -> irScanCount))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = new MatrixTable(hc, newMatrix).rowsTable().collect()
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx})
  }

  @Test def testScanCollectBehavesLikeRange() {
    val mt = getMatrixTable
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("range" -> irScanCollect(GetField(oldRow, "row_idx"))))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = new MatrixTable(hc, newMatrix).rowsTable().collect()
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, row_idx)})
  }


  @Test def testScanCollectBehavesLikeRangeWithAggregation() {
    val mt = getMatrixTable
    val oldRow = Ref("va", mt.typ.rvRowType)

    val newRow = InsertFields(oldRow, Seq("range" -> irScanCollect(irAggCount * GetField(oldRow, "row_idx").toL)))

    val newMatrix = MatrixMapRows(mt, newRow, None)
    val rows = new MatrixTable(hc, newMatrix).rowsTable().collect()
    val ranges = rows.map { case Row(_, range: IndexedSeq[Int]) => range }
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, row_idx).map(_ * 20)})
  }
}

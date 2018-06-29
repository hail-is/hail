package is.hail.expr

import is.hail.SparkSuite
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.MatrixTable
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class TableIRSuite extends SparkSuite {
  def getKT: Table = {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = IndexedSeq("Sample")

    val kt = Table(hc, rdd, signature, Some(keyNames))
    kt.typeCheck()
    kt
  }

  def rangeKT: TableIR = {
    //initialise hc
    hc.sc
    TableRange(20, 4)
  }

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

  @Test def testFilter() {
    val kt = getKT
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.GetField(ir.Ref("row", kt.typ.rowType), "field1"), ir.I32(3))))
    assert(kt2.count() == 1)
  }

  @Test def testFilterGlobals() {
    val kt = getKT.selectGlobal("{g: 3}")
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.GetField(ir.Ref("row", kt.typ.rowType), "field1"), ir.GetField(ir.Ref("global", kt.typ.globalType), "g"))))
    assert(kt2.count() == 1)
  }

  @Test def testScanCountBehavesLikeIndex() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("idx2" -> irScanCount))
    val newTable = TableMapRows(t, newRow, None, None)
    val rows = Interpret[IndexedSeq[Row]](TableAggregate(newTable, irAggCollect(Ref("row", newRow.typ))))
    assert(rows.forall { case Row(row_idx, idx) => row_idx == idx})
  }

  @Test def testScanCollectBehavesLikeRange() {
    val t = rangeKT
    val oldRow = Ref("row", t.typ.rowType)

    val newRow = InsertFields(oldRow, Seq("range" -> irScanCollect(GetField(oldRow, "idx"))))

    val newTable = TableMapRows(t, newRow, None, None)
    val rows = Interpret[IndexedSeq[Row]](TableAggregate(newTable, irAggCollect(Ref("row", newRow.typ))))
    assert(rows.forall { case Row(row_idx: Int, range: IndexedSeq[Int]) => range sameElements Array.range(0, row_idx)})
  }
}

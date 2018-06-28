package is.hail.expr

import is.hail.SparkSuite
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.table.Table
import is.hail.utils._
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

  @Test def testFilter() {
    val kt = getKT
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ApplyComparisonOp(EQ(TInt32()), GetField(Ref("row", kt.typ.rowType), "field1"), I32(3))))
    assert(kt2.count() == 1)
  }

  @Test def testFilterGlobals() {
    val kt = getKT.selectGlobal("{g: 3}")
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ApplyComparisonOp(EQ(TInt32()), GetField(Ref("row", kt.typ.rowType), "field1"), GetField(Ref("global", kt.typ.globalType), "g"))))
    assert(kt2.count() == 1)
  }

  @Test def testScan() {
    val kt = getKT
    val row = Ref("row", kt.typ.rowType)
    val aggSig = AggSignature(Count(), TInt32(), FastSeq.empty, None, FastSeq.empty)
    val newRow = InsertFields(row, Seq("idx" -> ApplyAggOp(SeqOp(0, 0, aggSig), FastIndexedSeq.empty, None, aggSig)))
    val indexWithScan = TableScan(kt.tir, newRow)
    val kt2 = new Table(hc, indexWithScan)
    assert(Array.range(0, kt2.count().toInt) sameElements kt2.collect().map { case Row(_, _, _, idx) => idx })
  }
}

package is.hail.expr

import is.hail.SparkSuite
import is.hail.expr.types._
import is.hail.table.Table
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
      ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.GetField(ir.Ref("row", kt.typ.rowType), "field1"), ir.I32(3))))
    assert(kt2.count() == 1)
  }

  @Test def testFilterGlobals() {
    val kt = getKT.selectGlobal("{g: 3}")
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ir.ApplyComparisonOp(ir.EQ(TInt32()), ir.GetField(ir.Ref("row", kt.typ.rowType), "field1"), ir.GetField(ir.Ref("global", kt.typ.globalType), "g"))))
    assert(kt2.count() == 1)
  }
}

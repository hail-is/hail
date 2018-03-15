package is.hail.expr

import is.hail.SparkSuite
import is.hail.expr.types._
import is.hail.table.Table
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class TableIRSuite extends SparkSuite {

  @Test def testFilter() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ir.ApplyBinaryPrimOp(ir.EQ(), ir.GetField(ir.Ref("row"), "field1"), ir.I32(3))))
    assert(kt2.count() == 1)
  }

  @Test def testFilterGlobals() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames).annotateGlobalExpr("g = 3")
    val kt2 = new Table(hc, TableFilter(kt.tir,
      ir.ApplyBinaryPrimOp(ir.EQ(), ir.GetField(ir.Ref("row"), "field1"), ir.GetField(ir.Ref("global"), "g"))))
    assert(kt2.count() == 1)
  }

  @Test def testAnnotate() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    val kt2 = new Table(hc, TableAnnotate(kt.tir,
      IndexedSeq("dummy", "a", "field1"),
      IndexedSeq(ir.I32(0), ir.GetField(ir.Ref("row"), "field1"), ir.GetField(ir.Ref("row"), "field2"))))
    assert(kt2.select("row.Sample", "field1 = row.a", "field2 = row.field1").same(kt))
  }
}

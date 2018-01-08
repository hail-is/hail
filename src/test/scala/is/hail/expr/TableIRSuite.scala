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
    val kt2 = new Table(hc, TableFilter(kt.ir, ir.ApplyBinaryPrimOp(ir.EQ(), ir.Ref("field1"), ir.I32(3))))
    assert(kt2.count() == 1)
  }

  @Test def testFilterGlobals() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames).annotateGlobalExpr("g = 3")
    val kt2 = new Table(hc, TableFilter(kt.ir, ir.ApplyBinaryPrimOp(ir.EQ(), ir.Ref("field1"), ir.Ref("g"))))
    assert(kt2.count() == 1)
  }

  @Test def testAnnotate() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)
    val kt2 = new Table(hc, TableAnnotate(kt.ir,
      IndexedSeq("a", "field1"),
      IndexedSeq(ir.I32(0), ir.InsertFields(ir.GetField(ir.In(0, kt.signature), "field1"), Array(("b", ir.GetField(ir.In(0, kt.signature), "field1")))))))
    assert(kt2.select("Sample", "field1 = field1.b", "field2").same(kt))
  }

  @Test def testAnnotate2() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = Table(hc, rdd, signature, keyNames)

    val kt2 = kt.annotate("a.b = field1, a.a = field2")
    assert(kt2.ir.isInstanceOf[TableAnnotate])

    val kt3 = kt2.annotate("a.c = 0")
    assert(kt3.ir.isInstanceOf[TableAnnotate])

    val kt4 = kt3.annotate("field1 = a")
    assert(kt4.ir.isInstanceOf[TableAnnotate])

    val kt5 = kt4.annotate("field2.b.c.d.e.f = field2")
    assert(kt5.ir.isInstanceOf[TableAnnotate])

    assert(kt2.select("Sample", "field1 = a.b", "field2 = a.a").same(kt))
    assert(kt3.select("Sample", "field1 = a.b", "field2 = a.a").same(kt))
    assert(kt4.select("Sample", "field1 = a.b", "field2 = a.a").same(kt))
    assert(kt4.select("Sample", "field1 = field1.b", "field2 = field1.a").same(kt))
    assert(kt5.select("Sample", "field1 = field1.b", "field2 = field2.b.c.d.e.f").same(kt))


  }
}

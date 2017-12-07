package is.hail.expr

import is.hail.SparkSuite
import is.hail.keytable.KeyTable
import is.hail.expr.ir
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class KeyTableIRSuite extends SparkSuite {

  @Test def testFilter() {
    val data = Array(Array("Sample1", 9, 5), Array("Sample2", 3, 5), Array("Sample3", 2, 5), Array("Sample4", 1, 5))
    val rdd = sc.parallelize(data.map(Row.fromSeq(_)))
    val signature = TStruct(("Sample", TString()), ("field1", TInt32()), ("field2", TInt32()))
    val keyNames = Array("Sample")

    val kt = KeyTable(hc, rdd, signature, keyNames)
    val kt2 = new KeyTable(hc, new FilterKT(kt.ir, ir.ApplyBinaryPrimOp(ir.EQ(), ir.Ref("field1"), ir.I32(3))))
    assert(kt2.count() == 1)
  }
}

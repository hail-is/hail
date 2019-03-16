package is.hail.io

import is.hail.{HailContext, SparkSuite}
import is.hail.annotations._
import is.hail.check.Gen
import is.hail.check.Prop.forAll
import is.hail.expr._
import is.hail.expr.types._
import is.hail.expr.types.virtual._
import is.hail.rvd.RVD
import is.hail.table.Table
import is.hail.variant.{MatrixTable, ReferenceGenome$, VSMSubgen}
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row

class ImportMatrixSuite extends SparkSuite {

  @Test def testHeadersNotIdentical() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0"))
    }
    assert(e.getMessage.contains("invalid header"))
  }

  @Test def testMissingVals() {
    val files = hc.hadoopConf.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = new MatrixTable(HailContext.get, LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0")))
      vsm.rvd.count()
    }
    assert(e.getMessage.contains("Incorrect number"))
  }
}

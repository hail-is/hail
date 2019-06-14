package is.hail.io

import is.hail.expr.types.virtual._
import is.hail.utils._
import is.hail.variant.MatrixTable
import is.hail.{HailContext, HailSuite}
import org.apache.spark.SparkException
import org.testng.annotations.Test

class ImportMatrixSuite extends HailSuite {

  @Test def testHeadersNotIdentical() {
    val files = hc.sFS.globAll(List("src/test/resources/sampleheader*.txt"))
    val e = intercept[SparkException] {
      val vsm = LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0"))
    }
    assert(e.getMessage.contains("invalid header"))
  }

  @Test def testMissingVals() {
    val files = hc.sFS.globAll(List("src/test/resources/samplesmissing.txt"))
    val e = intercept[SparkException] {
      val vsm = new MatrixTable(HailContext.get, LoadMatrix(hc, files, Map("f0" -> TString()), Array("f0")))
      vsm.rvd.count()
    }
    assert(e.getMessage.contains("Incorrect number"))
  }
}

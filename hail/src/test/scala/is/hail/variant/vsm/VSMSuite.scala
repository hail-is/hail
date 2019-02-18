package is.hail.variant.vsm

import is.hail.check.Parameters
import is.hail.expr.ir.{Interpret, MatrixIR, MatrixNativeWriter, MatrixRepartition, MatrixWrite, RepartitionStrategy, TableRepartition}
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant._
import is.hail.{SparkSuite, TestUtils}
import org.apache.commons.math3.stat.descriptive.SummaryStatistics
import org.apache.commons.math3.stat.regression.SimpleRegression
import org.testng.annotations.Test

import scala.language.postfixOps

class VSMSuite extends SparkSuite {
  @Test def testOverwrite() {
    def write(out: String, overwrite: Boolean = false) {
      Interpret(MatrixWrite(
        TestUtils.importVCF(hc, "src/test/resources/sample2.vcf"),
        MatrixNativeWriter(out, overwrite = overwrite)))
    }

    val out = tmpDir.createTempFile("out", "vds")
    write(out)

    TestUtils.interceptFatal("""file already exists""") {
      write(out)
    }

    write(out, overwrite = true)
  }

  @Test def testInvalidMetadata() {
    TestUtils.interceptFatal("metadata does not contain file version") {
      MatrixIR.read(hc, "src/test/resources/0.1-1fd5cc7.vds")
    }
  }

  @Test def testFilesWithRequiredGlobals() {
    val mt = MatrixIR.read(hc, "src/test/resources/required_globals.mt")
    Interpret(MatrixRepartition(mt, 10, RepartitionStrategy.SHUFFLE)).rvd.count()
  }
}

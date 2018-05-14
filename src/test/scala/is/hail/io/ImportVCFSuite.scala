package is.hail.io

import is.hail.check.Prop._
import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.io.vcf.{ExportVCF, LoadVCF}
import is.hail.variant._
import org.apache.spark.SparkException
import org.testng.annotations.Test
import is.hail.utils._
import is.hail.testUtils._
import org.apache.spark.sql.Row

class ImportVCFSuite extends SparkSuite {

  @Test def testParseCall() {
    import is.hail.io.vcf.HtsjdkRecordReader._
    assert(parseCall("0/0", 2) == Call2(0, 0))
    assert(parseCall("1/0", 2) == Call2(1, 0))
    assert(parseCall("0", 2) == Call1(0))
    assert(parseCall(".", 2) == null)
    assert(parseCall("./.", 2) == null)
    assert(parseCall("1|0", 2) == Call2(1, 0, phased = true))
    assert(parseCall("0|1", 2) == Call2(0, 1, phased = true))
    intercept[HailException] {
      parseCall("./0", 2) == Call2(0, 0)
    }
    intercept[HailException] {
      parseCall("""0\0""", 2) == Call2(0, 0)
    }
  }

  @Test def randomExportImportIsIdentity() {
    forAll(MatrixTable.gen(hc, VSMSubgen.random)) { vds =>

      val truth = {
        val f = tmpDir.createTempFile(extension="vcf")
        ExportVCF(vds, f)
        hc.importVCF(f, rg = Some(vds.referenceGenome))
      }

      val actual = {
        val f = tmpDir.createTempFile(extension="vcf")
        ExportVCF(truth, f)
        hc.importVCF(f, rg = Some(vds.referenceGenome))
      }

      truth.same(actual)
    }.check()
  }
}

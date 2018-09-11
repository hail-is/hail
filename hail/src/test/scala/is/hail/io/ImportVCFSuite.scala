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

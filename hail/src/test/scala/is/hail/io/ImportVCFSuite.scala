package is.hail.io

import is.hail.check.Prop._
import is.hail.expr.ir.ExecuteContext
import is.hail.io.vcf.ExportVCF
import is.hail.variant._
import is.hail.{HailSuite, TestUtils}
import org.testng.annotations.Test

class ImportVCFSuite extends HailSuite {
  @Test def randomExportImportIsIdentity() {
    ExecuteContext.scoped { ctx =>
      forAll(MatrixTable.gen(hc, VSMSubgen.random, ctx)) { vds =>

        val truth = {
          val f = tmpDir.createTempFile(extension="vcf")
          ExportVCF(vds, f)
          TestUtils.importVCF(hc, f, rg = Some(vds.referenceGenome))
        }

        val actual = {
          val f = tmpDir.createTempFile(extension="vcf")
          ExportVCF(truth, f)
          TestUtils.importVCF(hc, f, rg = Some(vds.referenceGenome))
        }

        truth.same(actual, ctx)
      }.check()
    }
  }
}

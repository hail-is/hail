package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test

class SelectSuite extends SparkSuite {
  @Test def testRows() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
      .annotateVariantsExpr("foo = gs.count()")

    val t1 = vds.selectRows("va.locus", "va.alleles", "va.info.AC", "AF = va.info.AF", "foo2 = gs.count()").rowsTable()

    val t2 = vds.rowsTable().select("locus", "alleles", "info.AC", "AF = info.AF", "foo2 = foo")

    assert(t1.same(t2))
  }

  @Test def testCols() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
      .annotateSamplesExpr("foo = gs.count(), bar.baz = 5")

    val t1 = vds.selectCols("sa.s", "sa.bar.baz", "foo2 = gs.count()").colsTable()

    val t2 = vds.colsTable().select("s", "bar.baz", "foo2 = foo")

    assert(t1.same(t2))
  }
}

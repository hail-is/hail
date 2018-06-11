package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test
import is.hail.testUtils._
import is.hail.utils.FastIndexedSeq

class SelectSuite extends SparkSuite {
  @Test def testRows() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
      .annotateRowsExpr("foo" -> "AGG.count()")

    val t1 = vds.selectRows("{locus: va.locus, alleles: va.alleles, AC: va.info.AC, AF: va.info.AF, foo2: AGG.count()}", None).rowsTable()

    val t2 = vds.rowsTable().select("{locus: row.locus, alleles: row.alleles, AC: row.info.AC, AF: row.info.AF, foo2: row.foo}",
      Some(FastIndexedSeq("locus", "alleles")), Some(2))

    assert(t1.same(t2))
  }

  @Test def test_key_change_typechecks() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")

    vds.selectRows("{alleles: va.alleles}", Some(IndexedSeq("alleles") -> IndexedSeq())).typecheck()
  }

  @Test def testCols() {
    val vds = hc.importVCF("src/test/resources/sample.vcf.bgz")
      .annotateColsExpr("foo" -> "AGG.count()", "bar" -> "{baz: 5}")

    val t1 = vds.selectCols("{s: sa.s, baz: sa.bar.baz, foo2: AGG.count()}", None).colsTable()

    val t2 = vds.colsTable().select("{s: row.s, baz: row.bar.baz, foo2: row.foo}", Option(FastIndexedSeq("s")), Option(1))

    assert(t1.same(t2))
  }
}

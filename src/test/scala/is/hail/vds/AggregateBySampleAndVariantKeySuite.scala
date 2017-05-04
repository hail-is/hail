package is.hail.vds

import is.hail.SparkSuite
import is.hail.utils._
import is.hail.expr.TDouble
import org.testng.annotations.Test

class AggregateBySampleAndVariantKeySuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.genes", all=true)
      .annotateVariantsExpr("va.weight = v.start.toDouble")
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno", "Sample", root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))

    val (_, sampleKT) = vds.linregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno")

    val kt = vds.aggregateBySampleAndVariantKey("gene", "va.genes", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()")

    assert(sampleKT same kt.select(sampleKT.fieldNames, sampleKT.keyNames))
  }
}

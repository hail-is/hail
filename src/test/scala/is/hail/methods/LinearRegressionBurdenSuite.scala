package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.TDouble
import is.hail.utils.TextTableConfiguration
import org.testng.annotations.Test

class LinearRegressionBurdenSuite extends SparkSuite {
  @Test def test() {

    val vds = hc.importVCF("src/test/resources/regressionLinearBurden.vcf")
      .filterMulti()
      .annotateVariantsIntervals("src/test/resources/regressionLinearBurden.interval_list", "va.gene", all=true)
      .filterVariantsExpr("! va.gene.isEmpty()")
      .annotateVariantsExpr("va.weight = v.start / 50.0")
      .variantQC()
      .annotateSamplesTable("src/test/resources/regressionLinearBurden.cov", "Sample", root = Some("sa"), config = TextTableConfiguration(types = Map("Pheno" -> TDouble, "Cov1" -> TDouble, "Cov2" -> TDouble)))

    val kt = vds.linregBurden("gene", "va.gene", "sum()", "va.weight * g.gt", "sa.Pheno", covSA = Array("sa.Cov1", "sa.Cov2"), dropSamples = false)

    kt.toDF(hc.sqlContext).show()
  }
}

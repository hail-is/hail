package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.TDouble
import is.hail.utils.TextTableConfiguration
import org.testng.annotations.Test

class LinearRegressionBurdenSuite extends SparkSuite {
  @Test def test() {

    hc.baldingNicholsModel(3, 10, 50, popDist = Some(Array(0.2, 0.3, 0.5)), fst = Some(Array(.05, .09, .11))).exportVCF("src/test/resources/regressionLinearBurden.vcf")

    val vds = hc.baldingNicholsModel(3, 10, 50, popDist = Some(Array(0.2, 0.3, 0.5)), fst = Some(Array(.05, .09, .11)))
      .annotateVariantsIntervals("/Users/jbloom/data/burden/genes.interval_list", "va.gene", all=true)
      .filterVariantsExpr("! va.gene.isEmpty()")
      .annotateVariantsExpr("va.weight = v.start / 50.0")
      .variantQC()
      .annotateSamplesTable("src/test/resources/regressionLinearBurden.cov", "Sample", root = Some("sa"), config = TextTableConfiguration(types = Map("Pheno" -> TDouble, "Cov1" -> TDouble, "Cov2" -> TDouble)))

    val kt = vds.linregBurden("gene", "va.gene", "sum()", "va.weight * g.gt", "sa.Pheno", covSA = Array("sa.Cov1", "sa.Cov2"), dropSamples = true)

    kt.toDF(hc.sqlContext).show()
  }
}

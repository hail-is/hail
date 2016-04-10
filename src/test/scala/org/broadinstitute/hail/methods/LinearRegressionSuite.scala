package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Querier
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

class LinearRegressionSuite extends SparkSuite {

  @Test def test() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("tsv",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamples.run(s, Array("tsv",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "--types", "Pheno: Double",
      "--missing", "0"))

    s = LinearRegressionCommand.run(s, Array(
      "-y", "sa.pheno.Pheno",
      "-c", "sa.cov.Cov1,sa.cov.Cov2",
      "-o", "/tmp/linearRegression.linreg"))

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qBeta = s.vds.queryVA("linreg", "beta")
    val qSe = s.vds.queryVA("linreg", "se")
    val qTstat = s.vds.queryVA("linreg", "tstat")
    val qPval = s.vds.queryVA("linreg", "pval")

    val annotationMap = s.vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertDouble(q: Querier, v: Variant, value: Double) =
      assert(D_==(q(annotationMap(v)).get.asInstanceOf[Double], value))

    def assertEmpty(q: Querier, v: Variant) =
      assert(q(annotationMap(v)).isEmpty)

    /*
    comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)
    */

    assertDouble(qBeta, v1, -0.28589421)
    assertDouble(qSe, v1,  1.2739153)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1,  0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta, v2, -0.5417647)
    assertDouble(qSe, v2, 0.3350599)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    assertEmpty(qBeta, v6)
    assertEmpty(qBeta, v7)
    assertEmpty(qBeta, v8)
    assertEmpty(qBeta, v9)
    assertEmpty(qBeta, v10)

    //val result = "rm -rf /tmp/linearRegression" !;
    //linreg.write("/tmp/linearRegression.linreg") //FIXME: How to test?

  }
}

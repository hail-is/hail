package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.TestUtils._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Querier
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant.Variant
import org.testng.annotations.Test

class LinearRegressionSuite extends SparkSuite {

  @Test def testWithTwoCov() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.cov",
      "-e", "Sample",
      "--root", "sa.cov",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "-e", "Sample",
      "--types", "Pheno: Double",
      "--missing", "0"))

    s = LinearRegressionCommand.run(s, Array(
      "-y", "sa.pheno.Pheno",
      "-c", "sa.cov.Cov1, sa.cov.Cov2 + 1 - 1"))

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v3 = Variant("1", 3, "C", "T")   // x = (0, ., 1, 1, 1, .)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qMissing = s.vds.queryVA("va.linreg.nMissing")._2
    val qBeta = s.vds.queryVA("va.linreg.beta")._2
    val qSe = s.vds.queryVA("va.linreg.se")._2
    val qTstat = s.vds.queryVA("va.linreg.tstat")._2
    val qPval = s.vds.queryVA("va.linreg.pval")._2

    val annotationMap = s.vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertInt(q: Querier, v: Variant, value: Int) =
      assert(D_==(q(annotationMap(v)).get.asInstanceOf[Int], value))

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
    summary(fit)["coefficients"]

    */

    assertInt(qMissing, v1, 0)
    assertDouble(qBeta, v1, -0.28589421)
    assertDouble(qSe, v1, 1.2739153)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1, 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertInt(qMissing, v2, 2)
    assertDouble(qBeta, v2, -0.5417647)
    assertDouble(qSe, v2, 0.3350599)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0, 0.75, 1, 1, 1, 0.75)
    */

    assertInt(qMissing, v3, 2)
    assertDouble(qBeta, v3, 1.07367185)
    assertDouble(qSe, v3, 0.6764348)
    assertDouble(qTstat, v3, 1.5872510)
    assertDouble(qPval, v3, 0.2533675)

    assertEmpty(qBeta, v6)
    assertEmpty(qBeta, v7)
    assertEmpty(qBeta, v8)
    assertEmpty(qBeta, v9)
    assertEmpty(qBeta, v10)
  }

  @Test def testWithNoCov() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "--types", "Pheno: Int",
      "-e", "Sample",
      "--missing", "0"))

    s = LinearRegressionCommand.run(s, Array(
      "-y", "sa.pheno.Pheno"))

    val v1 = Variant("1", 1, "C", "T") // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T") // x = (., 2, ., 2, 0, 0)
    val v6 = Variant("1", 6, "C", "T") // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T") // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T") // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T") // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qBeta = s.vds.queryVA("va.linreg.beta")._2
    val qSe = s.vds.queryVA("va.linreg.se")._2
    val qTstat = s.vds.queryVA("va.linreg.tstat")._2
    val qPval = s.vds.queryVA("va.linreg.pval")._2

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
    df = data.frame(y, x)
    fit <- lm(y ~ x, data=df)
    summary(fit)
    */

    assertDouble(qBeta, v1, -0.25)
    assertDouble(qSe, v1, 0.4841229)
    assertDouble(qTstat, v1, -0.5163978)
    assertDouble(qPval, v1, 0.63281250)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta, v2, -0.250000)
    assertDouble(qSe, v2, 0.2602082)
    assertDouble(qTstat, v2, -0.9607689)
    assertDouble(qPval, v2, 0.391075888)

    assertEmpty(qBeta, v6)
    assertEmpty(qBeta, v7)
    assertEmpty(qBeta, v8)
    assertEmpty(qBeta, v9)
    assertEmpty(qBeta, v10)
  }

  @Test def testWithImportFamBoolean() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "-e", "Sample",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamplesFam.run(s, Array(
      "-i", "src/test/resources/linearRegression.fam"))

    s = LinearRegressionCommand.run(s, Array(
      "-y", "sa.fam.isCase",
      "-c", "sa.cov.Cov1,sa.cov.Cov2"))

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val (_, qMissing) = s.vds.queryVA("va.linreg.nMissing")
    val (_, qBeta) = s.vds.queryVA("va.linreg.beta")
    val (_, qSe) = s.vds.queryVA("va.linreg.se")
    val (_, qTstat) = s.vds.queryVA("va.linreg.tstat")
    val (_, qPval) = s.vds.queryVA("va.linreg.pval")

    val annotationMap = s.vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertInt(q: Querier, v: Variant, value: Int) =
      assert(D_==(q(annotationMap(v)).get.asInstanceOf[Int], value))

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
    summary(fit)["coefficients"]

    */

    assertInt(qMissing, v1, 0)
    assertDouble(qBeta, v1, -0.28589421)
    assertDouble(qSe, v1, 1.2739153)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1, 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertInt(qMissing, v2, 2)
    assertDouble(qBeta, v2, -0.5417647)
    assertDouble(qSe, v2, 0.3350599)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    assertEmpty(qBeta, v6)
    assertEmpty(qBeta, v7)
    assertEmpty(qBeta, v8)
    assertEmpty(qBeta, v9)
    assertEmpty(qBeta, v10)
  }

  @Test def testWithImportFam() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "-e", "Sample",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamplesFam.run(s, Array(
      "-i", "src/test/resources/linearRegression.fam",
      "-q",
      "-m", "0"))

    s = LinearRegressionCommand.run(s, Array(
      "-y", "sa.fam.qPheno",
      "-c", "sa.cov.Cov1,sa.cov.Cov2"))

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qMissing = s.vds.queryVA("va.linreg.nMissing")._2
    val qBeta = s.vds.queryVA("va.linreg.beta")._2
    val qSe = s.vds.queryVA("va.linreg.se")._2
    val qTstat = s.vds.queryVA("va.linreg.tstat")._2
    val qPval = s.vds.queryVA("va.linreg.pval")._2

    val annotationMap = s.vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertInt(q: Querier, v: Variant, value: Int) =
      assert(D_==(q(annotationMap(v)).get.asInstanceOf[Int], value))

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
    summary(fit)["coefficients"]

    */

    assertInt(qMissing, v1, 0)
    assertDouble(qBeta, v1, -0.28589421)
    assertDouble(qSe, v1, 1.2739153)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1, 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertInt(qMissing, v2, 2)
    assertDouble(qBeta, v2, -0.5417647)
    assertDouble(qSe, v2, 0.3350599)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    assertEmpty(qBeta, v6)
    assertEmpty(qBeta, v7)
    assertEmpty(qBeta, v8)
    assertEmpty(qBeta, v9)
    assertEmpty(qBeta, v10)
  }

  @Test def testNonNumericPheno() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "-e", "Sample",
      "--types", "Cov1: Double, Cov2: Double"))

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "-e", "Sample",
      "--types", "Pheno: String",
      "--missing", "0"))

    interceptFatal("Sample annotation `sa.pheno.Pheno' must be numeric or Boolean, got String") {
      LinearRegressionCommand.run(s, Array(
        "-y", "sa.pheno.Pheno",
        "-c", "sa.cov.Cov1,sa.cov.Cov2"))
    }
  }

  @Test def testNonNumericCov() {
    var s = State(sc, sqlContext)

    s = ImportVCF.run(s, Array("src/test/resources/linearRegression.vcf"))

    s = SplitMulti.run(s)

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.cov",
      "--root", "sa.cov",
      "-e", "Sample",
      "--types", "Cov1: Double, Cov2: String"))

    s = AnnotateSamples.run(s, Array("table",
      "-i", "src/test/resources/linearRegression.pheno",
      "--root", "sa.pheno",
      "-e", "Sample",
      "--types", "Pheno: Double",
      "--missing", "0"))

    interceptFatal("Sample annotation `sa.cov.Cov2' must be numeric or Boolean, got String") {
      LinearRegressionCommand.run(s, Array(
        "-y", "sa.pheno.Pheno",
        "-c", "sa.cov.Cov1,sa.cov.Cov2"))
    }
  }
}

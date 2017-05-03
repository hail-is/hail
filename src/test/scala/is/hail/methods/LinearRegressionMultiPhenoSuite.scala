package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Querier
import is.hail.expr.TDouble
import is.hail.utils._
import is.hail.variant.{Genotype, Variant}
import org.testng.annotations.Test

class LinearRegressionMultiPhenoSuite extends SparkSuite {

  // ensuring that result for one phenotype and hard calls is the same as with linreg
  // and that including the same phenotype a second time gives same result as well.
  @Test def testWithTwoCov() {
    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .linregMultiPheno(Array("sa.pheno.Pheno", "sa.pheno.Pheno"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v3 = Variant("1", 3, "C", "T")   // x = (0, ., 1, 1, 1, .)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val annotationMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertInt(q: Querier, v: Variant, value: Int) =
      assert(D_==(q(annotationMap(v)).asInstanceOf[Int], value))

    def assertDouble(q: Querier, v: Variant, value: Double) = {
      val x = q(annotationMap(v)).asInstanceOf[IndexedSeq[Double]].apply(0)
      assert(D_==(x, value))
    }

    def assertDouble1(q: Querier, v: Variant, value: Double) = {
      val x = q(annotationMap(v)).asInstanceOf[IndexedSeq[Double]].apply(1)
      assert(D_==(x, value))
    }

    def assertEmpty(q: Querier, v: Variant) =
      assert(q(annotationMap(v)) == null)

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

    assertDouble(qBeta, v1, -0.28589421)
    assertDouble(qSe, v1, 1.2739153)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1, 0.84327106)

    // checking that second copy of phenotype gives the same answer
    assertDouble1(qBeta, v1, -0.28589421)
    assertDouble1(qSe, v1, 1.2739153)
    assertDouble1(qTstat, v1, -0.22442167)
    assertDouble1(qPval, v1, 0.84327106)


    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta, v2, -0.5417647)
    assertDouble(qSe, v2, 0.3350599)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0, 0.75, 1, 1, 1, 0.75)
    */

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

  // ensuring that result for one phenotype and dosages is the same as with linreg.
  @Test def testWithTwoCovPhred() {
    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .linregMultiPheno(Array("sa.pheno.Pheno"), Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages = true)

    val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
    val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
    val v3 = Variant("1", 3, "C", "T")   // x = (0, ., 1, 1, 1, .)
    val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
    val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
    val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
    val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
    val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val annotationMap = vds.variantsAndAnnotations
      .collect()
      .toMap

    def assertInt(q: Querier, v: Variant, value: Int) =
      assert(D_==(q(annotationMap(v)).asInstanceOf[Int], value))

    def assertDouble(q: Querier, v: Variant, value: Double) = {
      val x = q(annotationMap(v)).asInstanceOf[IndexedSeq[Double]].apply(0)
      assert(D_==(x, value))
    }

    def assertEmpty(q: Querier, v: Variant) =
      assert(q(annotationMap(v)) == null)

    /*
    comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0.009900990296049406, 0.9900990100009803, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.9900990100009803)
    c1 = c(0, 2, 1, -2, -2, 4)
    c2 = c(-1, 3, 5, 0, -4, 3)
    df = data.frame(y, x, c1, c2)
    fit <- lm(y ~ x + c1 + c2, data=df)
    summary(fit)["coefficients"]

    */

    assertDouble(qBeta, v1, -0.29166985)
    assertDouble(qSe, v1, 1.2996510)
    assertDouble(qTstat, v1, -0.22442167)
    assertDouble(qPval, v1, 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.9950495050004902, 1.980198019704931, 0.9950495050004902, 1.980198019704931, 0.009900990296049406, 0.009900990296049406)
    */

    assertDouble(qBeta, v2, -0.5499320)
    assertDouble(qSe, v2, 0.3401110)
    assertDouble(qTstat, v2, -1.616919)
    assertDouble(qPval, v2, 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.009900990296049406, 0.7450495050747477, 0.9900990100009803, 0.9900990100009803, 0.9900990100009803, 0.7450495050747477)
    */

    assertDouble(qBeta, v3, 1.09536219)
    assertDouble(qSe, v3, 0.6901002)
    assertDouble(qTstat, v3, 1.5872510)
    assertDouble(qPval, v3, 0.2533675)

    assertEmpty(qBeta, v6)
  }
}
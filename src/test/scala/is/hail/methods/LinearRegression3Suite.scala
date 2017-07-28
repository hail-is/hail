package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.expr.TDouble
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.testng.annotations.Test

class LinearRegression3Suite extends SparkSuite {
  def assertDouble(a: Annotation, value: Double, i: Int = 0) { 
    assert(D_==(a.asInstanceOf[IndexedSeq[Double]].apply(i), value)) 
  }
  def isNaN(a: Annotation): Boolean = a.asInstanceOf[IndexedSeq[Double]].apply(0).isNaN
  def isNearly1(a: Annotation): Boolean = D_==(a.asInstanceOf[IndexedSeq[Double]].apply(0), 1.0)
  def assertSingular(a: Annotation) { assert(a == null || isNaN(a) || isNearly1(a)) }
  
  val covariates: KeyTable = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)).keyBy("Sample")
  val phenotypes: KeyTable = hc.importTable("src/test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TDouble), missing = "0").keyBy("Sample")
  val vds0: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateSamplesTable(covariates, root = "sa.cov")
    .annotateSamplesTable(phenotypes, root = "sa.pheno")
  
  val v1 = Variant("1", 1, "C", "T")   // x = (0, 1, 0, 0, 0, 1)
  val v2 = Variant("1", 2, "C", "T")   // x = (., 2, ., 2, 0, 0)
  val v3 = Variant("1", 3, "C", "T")   // x = (0, ., 1, 1, 1, .)
  val v6 = Variant("1", 6, "C", "T")   // x = (0, 0, 0, 0, 0, 0)
  val v7 = Variant("1", 7, "C", "T")   // x = (1, 1, 1, 1, 1, 1)
  val v8 = Variant("1", 8, "C", "T")   // x = (2, 2, 2, 2, 2, 2)
  val v9 = Variant("1", 9, "C", "T")   // x = (., 1, 1, 1, 1, 1)
  val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

  // ensuring that result for one phenotype and hard calls is the same as with linreg
  // and that including the same phenotype a second time gives same result as well.
  @Test def testWithTwoCov() {

    val vds = vds0.linreg3(Array("sa.pheno", "sa.pheno"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val am = vds.variantsAndAnnotations.collect().toMap

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
    assertDouble(qBeta(am(v1)), -0.28589421)
    assertDouble(qSe(am(v1)), 1.2739153)
    assertDouble(qTstat(am(v1)), -0.22442167)
    assertDouble(qPval(am(v1)), 0.84327106)

    // checking that second copy of phenotype gives the same answer
    assertDouble(qBeta(am(v1)), -0.28589421, 1)
    assertDouble(qSe(am(v1)), 1.2739153)
    assertDouble(qTstat(am(v1)), -0.22442167)
    assertDouble(qPval(am(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */
    assertDouble(qBeta(am(v2)), -0.5417647)
    assertDouble(qSe(am(v2)), 0.3350599)
    assertDouble(qTstat(am(v2)), -1.616919)
    assertDouble(qPval(am(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0, 0.75, 1, 1, 1, 0.75)
    */
    assertDouble(qBeta(am(v3)), 1.07367185)
    assertDouble(qSe(am(v3)), 0.6764348)
    assertDouble(qTstat(am(v3)), 1.5872510)
    assertDouble(qPval(am(v3)), 0.2533675)
  }

  // ensuring that result for one phenotype and dosages is the same as with linreg.
  @Test def testWithTwoCovPhred() {
    val vds = vds0.linreg3(Array("sa.pheno"), Array("sa.cov.Cov1", "sa.cov.Cov2"), useDosages = true)

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val am = vds.variantsAndAnnotations.collect().toMap

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
    assertDouble(qBeta(am(v1)), -0.29166985)
    assertDouble(qSe(am(v1)), 1.2996510)
    assertDouble(qTstat(am(v1)), -0.22442167)
    assertDouble(qPval(am(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.9950495050004902, 1.980198019704931, 0.9950495050004902, 1.980198019704931, 0.009900990296049406, 0.009900990296049406)
    */
    assertDouble(qBeta(am(v2)), -0.5499320)
    assertDouble(qSe(am(v2)), 0.3401110)
    assertDouble(qTstat(am(v2)), -1.616919)
    assertDouble(qPval(am(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.009900990296049406, 0.7450495050747477, 0.9900990100009803, 0.9900990100009803, 0.9900990100009803, 0.7450495050747477)
    */
    assertDouble(qBeta(am(v3)), 1.09536219)
    assertDouble(qSe(am(v3)), 0.6901002)
    assertDouble(qTstat(am(v3)), 1.5872510)
    assertDouble(qPval(am(v3)), 0.2533675)
  }

  @Test def testBgen12FastPath() {
    hc.indexBgen("src/test/resources/example.8bits.bgen")
    val vds = hc.importBgen("src/test/resources/example.8bits.bgen")
      // linreg3_sa has random normal pheno1, pheno2, cov1, cov2 with a small amount of missingness
      .annotateSamplesTable(hc.importTable("src/test/resources/linreg3_sa.tsv", keyNames = Array("Sample"), impute = true), root = "sa")
      .linreg3(Array("sa.pheno1", "sa.pheno2"), Array("sa.cov1", "sa.cov2"), useDosages = true, root = "va.linreg3")
      .linregMultiPheno(Array("sa.pheno1", "sa.pheno2"), Array("sa.cov1", "sa.cov2"), useDosages = true, root = "va.multi")
      // lingreg3 is defined and agrees with multi when it is defined
      .annotateVariantsExpr(
        "va.close = isMissing(va.multi) || (va.linreg3.beta - va.multi.beta).forall(d => d < 0.001)")

    val (_, q) = vds.queryVA("va.close")

    vds.rdd.foreach { case (v, (va, gs)) =>
      val a = q(va)
      simpleAssert(a != null && a.asInstanceOf[Boolean])
    }
  }
}

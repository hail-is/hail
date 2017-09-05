package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils._
import is.hail.annotations.Annotation
import is.hail.expr.{TFloat64, TString}
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.testng.annotations.Test

class LinearRegressionSuite extends SparkSuite {
  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6, fieldIndex: Int = 0, phenoIndex: Int = 0 ) {
    assert(D_==(a.asInstanceOf[IndexedSeq[IndexedSeq[Double]]].apply(fieldIndex)(phenoIndex), value, tol))
  }

  def assertNaN(a: Annotation, fieldIndex: Int = 0, phenoIndex: Int = 0) {
    assert(a.asInstanceOf[IndexedSeq[IndexedSeq[Double]]].apply(fieldIndex)(phenoIndex).isNaN)
  }
  
  def assertConsistentWithConstant(a: Annotation, fieldIndex: Int = 0, phenoIndex: Int = 0) {
    val ad = a.asInstanceOf[IndexedSeq[IndexedSeq[Double]]].apply(fieldIndex)(phenoIndex)
    assert(ad.isNaN || ad.asInstanceOf[Double] > 0.9999)
  }

  val v1 = Variant("1", 1, "C", "T") // x = (0, 1, 0, 0, 0, 1)
  val v2 = Variant("1", 2, "C", "T") // x = (., 2, ., 2, 0, 0)
  val v3 = Variant("1", 3, "C", "T") // x = (0, ., 1, 1, 1, .)
  val v6 = Variant("1", 6, "C", "T") // x = (0, 0, 0, 0, 0, 0)
  val v7 = Variant("1", 7, "C", "T") // x = (1, 1, 1, 1, 1, 1)
  val v8 = Variant("1", 8, "C", "T") // x = (2, 2, 2, 2, 2, 2)
  val v9 = Variant("1", 9, "C", "T") // x = (., 1, 1, 1, 1, 1)
  val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)
  
  val phenoKT: KeyTable = hc.importTable("src/test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TFloat64), missing = "0").keyBy("Sample")
  val covsKT: KeyTable = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TFloat64, "Cov2" -> TFloat64)).keyBy("Sample")
  val vds0: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateSamplesTable(phenoKT, root = "sa.pheno")
    .annotateSamplesTable(covsKT, root = "sa.cov")


  @Test def testWithTwoCov() {
    val vds = vds0.linreg(Array("sa.pheno"), Array("g.gt"), Array("sa.cov.Cov1", "sa.cov.Cov2 + 1 - 1"))

    val am = vds.variantsAndAnnotations.collect().toMap

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

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

    assertNaN(qSe(am(v6)))
    assertNaN(qTstat(am(v6)))
    assertNaN(qPval(am(v6)))

    assertNaN(qSe(am(v7)))
    assertNaN(qSe(am(v8)))
    assertNaN(qSe(am(v9)))
    assertNaN(qSe(am(v10)))
  }

  @Test def testWithTwoCovPhred() {
    val vds = vds0.linreg(Array("sa.pheno"), Array("g.dosage"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

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
        
    assertConsistentWithConstant(qPval(am(v6)))
  }

  @Test def testWithNoCov() {
    val vds = vds0.linreg(Array("sa.pheno"), Array("g.gt"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val am = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    df = data.frame(y, x)
    fit <- lm(y ~ x, data=df)
    summary(fit)
    */
    assertDouble(qBeta(am(v1)), -0.25)
    assertDouble(qSe(am(v1)), 0.4841229)
    assertDouble(qTstat(am(v1)), -0.5163978)
    assertDouble(qPval(am(v1)), 0.63281250)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */
    assertDouble(qBeta(am(v2)), -0.250000)
    assertDouble(qSe(am(v2)), 0.2602082)
    assertDouble(qTstat(am(v2)), -0.9607689)
    assertDouble(qPval(am(v2)), 0.391075888)

    assertNaN(qSe(am(v6)))
    assertNaN(qSe(am(v7)))
    assertNaN(qSe(am(v8)))
    assertNaN(qSe(am(v9)))
    assertNaN(qSe(am(v10)))
  }

  @Test def testWithImportFamBoolean() {
    val vds = vds0
      .annotateSamplesTable(KeyTable.importFam(hc, "src/test/resources/regressionLinear.fam"), root = "sa.fam")
      .linreg(Array("sa.fam.isCase"), Array("g.gt"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

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

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */
    assertDouble(qBeta(am(v2)), -0.5417647)
    assertDouble(qSe(am(v2)), 0.3350599)
    assertDouble(qTstat(am(v2)), -1.616919)
    assertDouble(qPval(am(v2)), 0.24728705)

    assertNaN(qSe(am(v6)))
    assertNaN(qSe(am(v7)))
    assertNaN(qSe(am(v8)))
    assertNaN(qSe(am(v9)))
    assertNaN(qSe(am(v10)))
  }

  @Test def testWithImportFam() {
    val vds = vds0
      .annotateSamplesTable(KeyTable.importFam(hc, "src/test/resources/regressionLinear.fam", isQuantitative = true, missingValue = "0"), root = "sa.fam")
      .linreg(Array("sa.fam.qPheno"), Array("g.gt"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

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

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */
    assertDouble(qBeta(am(v2)), -0.5417647)
    assertDouble(qSe(am(v2)), 0.3350599)
    assertDouble(qTstat(am(v2)), -1.616919)
    assertDouble(qPval(am(v2)), 0.24728705)

    assertNaN(qSe(am(v6)))
    assertNaN(qSe(am(v7)))
    assertNaN(qSe(am(v8)))
    assertNaN(qSe(am(v9)))
    assertNaN(qSe(am(v10)))
  }

  @Test def testNonNumericFatals() {
    val phenoStrKT = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TString), missing = "0").keyBy("Sample")
    val covsStrKT = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TString, "Cov2" -> TString)).keyBy("Sample")
    val vds = vds0.annotateSamplesTable(phenoStrKT, root = "sa.phenoStr")
                  .annotateSamplesTable(covsStrKT, root = "sa.covStr")
    
    interceptFatal("Sample annotation `sa.phenoStr' must be numeric or Boolean, got String") {
      vds.linreg(Array("sa.phenoStr"), Array("g.gt"))
    }
    
    interceptFatal("Sample annotation `sa.covStr.Cov1' must be numeric or Boolean, got String") {
      vds.linreg(Array("sa.pheno"), Array("sa.covStr.Cov1"))
    }
  }

  @Test def testMultiPhenoSame() {
    for (i <- Seq(0, 1);
      g <- Seq("g.gt", "g.dosage")) {
      val result = vds0
        .linreg(Array("sa.pheno"), Array(g), Array("sa.cov.Cov1", "sa.cov.Cov2"))
        .annotateVariantsVDS(
          vds0.linreg(Array("sa.pheno", "sa.pheno"), Array(g), Array("sa.cov.Cov1", "sa.cov.Cov2"))
            .annotateVariantsExpr(
              s"""
                 |va.linreg.beta = [[va.linreg.beta[0][$i]]],
                 |va.linreg.se = [[va.linreg.se[0][$i]]],
                 |va.linreg.tstat = [[va.linreg.tstat[0][$i]]],
                 |va.linreg.pval = [[va.linreg.pval[0][$i]]]
                 |""".stripMargin),
          code = Some("va.mlinreg = vds.linreg"))

      val (t, q) = result.queryVA("va.linreg")
      val (mt, mq) = result.queryVA("va.mlinreg")
      assert(t == mt)

      result.variantsAndAnnotations.collect().foreach { case (v, va) =>
        assert(t.valuesSimilar(q(va), mq(va)))
      }
    }
  }
  
  def isNaN(a: Annotation): Boolean = a.asInstanceOf[IndexedSeq[IndexedSeq[Double]]].apply(0)(0).isNaN
  def isNearly1(a: Annotation): Boolean = D_==(a.asInstanceOf[IndexedSeq[IndexedSeq[Double]]].apply(0)(0), 1.0)
  def assertSingular(a: Annotation) { assert(a == null || isNaN(a) || isNearly1(a)) }
  
  // ensuring that result for one phenotype and two fields is the same as with linreg
  @Test def testWithOneCovTwoFields() {
    val vds = vds0.linreg(Array("sa.pheno"), Array("g.gt", "sa.cov.Cov2"), Array("sa.cov.Cov1"))

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

    // checking coefficients of gt
    assertDouble(qBeta(am(v1)), -0.28589421)
    assertDouble(qSe(am(v1)), 1.2739153)
    assertDouble(qTstat(am(v1)), -0.22442167)
    assertDouble(qPval(am(v1)), 0.84327106)

    // checking coefficients of cov2
    assertDouble(qBeta(am(v1)), 0.02770781, fieldIndex = 1)
    assertDouble(qSe(am(v1)), 0.1643444, fieldIndex = 1)
    assertDouble(qTstat(am(v1)), 0.16859599, fieldIndex = 1)
    assertDouble(qPval(am(v1)), 0.88162287, fieldIndex = 1)
    
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
    
    assertSingular(qSe(am(v6)))
    assertSingular(qSe(am(v7)))
    assertSingular(qSe(am(v8)))
    assertSingular(qSe(am(v9)))
    assertSingular(qSe(am(v10)))
  }
  
  /* This test requires lifting linreg to run on GDS in subsequent PR
  @Test def testWithTwoCovDosage() {
    // .gen and .sample files created from regressionLinear.vcf
    // dosages are derived from PLs so results should agree with testWithTwoCovPhred
    val vds = hc.importGen("src/test/resources/regressionLinear.gen", "src/test/resources/regressionLinear.sample")
      .annotateSamplesTable(covsKT, root = "sa.cov")
      .annotateSamplesTable(phenoKT, root = "sa.pheno")
      .linreg(Array("sa.pheno"), Array("g.dosage"), Array("sa.cov.Cov1", "sa.cov.Cov2"))

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

    assertDouble(qBeta(am(v1)), -0.29166985, 1e-4)
    assertDouble(qSe(am(v1)), 1.2996510, 1e-4)
    assertDouble(qTstat(am(v1)), -0.22442167)
    assertDouble(qPval(am(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.9950495050004902, 1.980198019704931, 0.9950495050004902, 1.980198019704931, 0.009900990296049406, 0.009900990296049406)
    */
    assertDouble(qBeta(am(v2)), -0.5499320, 1e-4)
    assertDouble(qSe(am(v2)), 0.3401110, 1e-4)
    assertDouble(qTstat(am(v2)), -1.616919)
    assertDouble(qPval(am(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.009900990296049406, 0.7450495050747477, 0.9900990100009803, 0.9900990100009803, 0.9900990100009803, 0.7450495050747477)
    */
    assertDouble(qBeta(am(v3)), 1.09536219, 1e-4)
    assertDouble(qSe(am(v3)), 0.6901002, 1e-4)
    assertDouble(qTstat(am(v3)), 1.5872510)
    assertDouble(qPval(am(v3)), 0.2533675)

    assertNaN(qSe(am(v6)))
  }
  */
}

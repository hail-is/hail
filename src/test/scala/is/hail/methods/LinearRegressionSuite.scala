package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils._
import is.hail.annotations.Annotation
import is.hail.expr.typ.TString
import is.hail.table.Table
import is.hail.utils._
import is.hail.variant.Variant
import org.testng.annotations.Test

class LinearRegressionSuite extends SparkSuite {

  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
    assert(D_==(a.asInstanceOf[IndexedSeq[Double]].apply(0), value, tol))
  }

  def assertNaN(a: Annotation) {
    assert(a.asInstanceOf[IndexedSeq[Double]].apply(0).isNaN)
  }

  val v1 = Variant("1", 1, "C", "T") // x = (0, 1, 0, 0, 0, 1)
  val v2 = Variant("1", 2, "C", "T") // x = (., 2, ., 2, 0, 0)
  val v3 = Variant("1", 3, "C", "T") // x = (0, ., 1, 1, 1, .)
  val v6 = Variant("1", 6, "C", "T") // x = (0, 0, 0, 0, 0, 0)
  val v7 = Variant("1", 7, "C", "T") // x = (1, 1, 1, 1, 1, 1)
  val v8 = Variant("1", 8, "C", "T") // x = (2, 2, 2, 2, 2, 2)
  val v9 = Variant("1", 9, "C", "T") // x = (., 1, 1, 1, 1, 1)
  val v10 = Variant("1", 10, "C", "T") // x = (., 2, 2, 2, 2, 2)

  @Test def testWithTwoCov() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .linreg(Array("sa.pheno"), "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2 + 1 - 1"))
    
    val a = vds.variantsAndAnnotations.collect().toMap

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

    assertDouble(qBeta(a(v1)), -0.28589421)
    assertDouble(qSe(a(v1)), 1.2739153)
    assertDouble(qTstat(a(v1)), -0.22442167)
    assertDouble(qPval(a(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta(a(v2)), -0.5417647)
    assertDouble(qSe(a(v2)), 0.3350599)
    assertDouble(qTstat(a(v2)), -1.616919)
    assertDouble(qPval(a(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0, 0.75, 1, 1, 1, 0.75)
    */

    assertDouble(qBeta(a(v3)), 1.07367185)
    assertDouble(qSe(a(v3)), 0.6764348)
    assertDouble(qTstat(a(v3)), 1.5872510)
    assertDouble(qPval(a(v3)), 0.2533675)

    assertNaN(qSe(a(v6)))
    assertNaN(qTstat(a(v6)))
    assertNaN(qPval(a(v6)))

    assertNaN(qSe(a(v7)))
    assertNaN(qSe(a(v8)))
    assertNaN(qSe(a(v9)))
    assertNaN(qSe(a(v10)))
  }

  @Test def testWithTwoCovPhred() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .linreg(Array("sa.pheno"), "plDosage(g.PL)", Array("sa.cov.Cov1", "sa.cov.Cov2 + 1 - 1"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

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

    assertDouble(qBeta(a(v1)), -0.29166985)
    assertDouble(qSe(a(v1)), 1.2996510)
    assertDouble(qTstat(a(v1)), -0.22442167)
    assertDouble(qPval(a(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.9950495050004902, 1.980198019704931, 0.9950495050004902, 1.980198019704931, 0.009900990296049406, 0.009900990296049406)
    */

    assertDouble(qBeta(a(v2)), -0.5499320)
    assertDouble(qSe(a(v2)), 0.3401110)
    assertDouble(qTstat(a(v2)), -1.616919)
    assertDouble(qPval(a(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.009900990296049406, 0.7450495050747477, 0.9900990100009803, 0.9900990100009803, 0.9900990100009803, 0.7450495050747477)
    */

    assertDouble(qBeta(a(v3)), 1.09536219)
    assertDouble(qSe(a(v3)), 0.6901002)
    assertDouble(qTstat(a(v3)), 1.5872510)
    assertDouble(qPval(a(v3)), 0.2533675)
  }

  @Test def testWithTwoCovDosage() {
    // .gen and .sample files created from regressionLinear.vcf
    // dosages are derived from PLs so results should agree with testWithTwoCovPhred
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importGen("src/test/resources/regressionLinear.gen", "src/test/resources/regressionLinear.sample")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .linreg(Array("sa.pheno"), "dosage(g.GP)", Array("sa.cov.Cov1", "sa.cov.Cov2 + 1 - 1"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

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

    assertDouble(qBeta(a(v1)), -0.29166985, 1e-4)
    assertDouble(qSe(a(v1)), 1.2996510, 1e-4)
    assertDouble(qTstat(a(v1)), -0.22442167)
    assertDouble(qPval(a(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.9950495050004902, 1.980198019704931, 0.9950495050004902, 1.980198019704931, 0.009900990296049406, 0.009900990296049406)
    */

    assertDouble(qBeta(a(v2)), -0.5499320, 1e-4)
    assertDouble(qSe(a(v2)), 0.3401110, 1e-4)
    assertDouble(qTstat(a(v2)), -1.616919)
    assertDouble(qPval(a(v2)), 0.24728705)

    /*
    v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(0.009900990296049406, 0.7450495050747477, 0.9900990100009803, 0.9900990100009803, 0.9900990100009803, 0.7450495050747477)
    */

    assertDouble(qBeta(a(v3)), 1.09536219, 1e-4)
    assertDouble(qSe(a(v3)), 0.6901002, 1e-4)
    assertDouble(qTstat(a(v3)), 1.5872510)
    assertDouble(qPval(a(v3)), 0.2533675)

    assertNaN(qSe(a(v6)))
  }

  @Test def testWithNoCov() {
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .linreg(Array("sa.pheno"), "g.GT.nNonRefAlleles()", Array.empty[String])

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    y = c(1, 1, 2, 2, 2, 2)
    x = c(0, 1, 0, 0, 0, 1)
    df = data.frame(y, x)
    fit <- lm(y ~ x, data=df)
    summary(fit)
    */

    assertDouble(qBeta(a(v1)), -0.25)
    assertDouble(qSe(a(v1)), 0.4841229)
    assertDouble(qTstat(a(v1)), -0.5163978)
    assertDouble(qPval(a(v1)), 0.63281250)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta(a(v2)), -0.250000)
    assertDouble(qSe(a(v2)), 0.2602082)
    assertDouble(qTstat(a(v2)), -0.9607689)
    assertDouble(qPval(a(v2)), 0.391075888)

    assertNaN(qSe(a(v6)))
    assertNaN(qSe(a(v7)))
    assertNaN(qSe(a(v8)))
    assertNaN(qSe(a(v9)))
    assertNaN(qSe(a(v10)))
  }

  @Test def testWithImportFamBoolean() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(Table.importFam(hc, "src/test/resources/regressionLinear.fam"), root = "sa.fam")
      .linreg(Array("sa.fam.isCase"), "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

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

    assertDouble(qBeta(a(v1)), -0.28589421)
    assertDouble(qSe(a(v1)), 1.2739153)
    assertDouble(qTstat(a(v1)), -0.22442167)
    assertDouble(qPval(a(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta(a(v2)), -0.5417647)
    assertDouble(qSe(a(v2)), 0.3350599)
    assertDouble(qTstat(a(v2)), -1.616919)
    assertDouble(qPval(a(v2)), 0.24728705)

    assertNaN(qSe(a(v6)))
    assertNaN(qSe(a(v7)))
    assertNaN(qSe(a(v8)))
    assertNaN(qSe(a(v9)))
    assertNaN(qSe(a(v10)))
  }

  @Test def testWithImportFam() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(Table.importFam(hc, "src/test/resources/regressionLinear.fam", isQuantitative = true, missingValue = "0"), root = "sa.fam")
      .linreg(Array("sa.fam.qPheno"), "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.linreg.beta")._2
    val qSe = vds.queryVA("va.linreg.se")._2
    val qTstat = vds.queryVA("va.linreg.tstat")._2
    val qPval = vds.queryVA("va.linreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

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

    assertDouble(qBeta(a(v1)), -0.28589421)
    assertDouble(qSe(a(v1)), 1.2739153)
    assertDouble(qTstat(a(v1)), -0.22442167)
    assertDouble(qPval(a(v1)), 0.84327106)

    /*
    v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    x = c(1, 2, 1, 2, 0, 0)
    */

    assertDouble(qBeta(a(v2)), -0.5417647)
    assertDouble(qSe(a(v2)), 0.3350599)
    assertDouble(qTstat(a(v2)), -1.616919)
    assertDouble(qPval(a(v2)), 0.24728705)

    assertNaN(qSe(a(v6)))
    assertNaN(qSe(a(v7)))
    assertNaN(qSe(a(v8)))
    assertNaN(qSe(a(v9)))
    assertNaN(qSe(a(v10)))
  }

  @Test def testNonNumericPheno() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TString()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")

    interceptFatal("`sa.pheno' must be numeric or Boolean, got String") {
      vds.linreg(Array("sa.pheno"), "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }
  }

  @Test def testNonNumericCov() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TString())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")

    interceptFatal("`sa.cov.Cov2' must be numeric or Boolean, got String") {
      vds.linreg(Array("sa.pheno"), "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }
  }

  @Test def testMultiPhenoSame() {
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val inputVDS = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")

    for (i <- Seq(0, 1);
      d <- Seq(false, true)) {
      val result = inputVDS
        .linreg(Array("sa.pheno"),
          if (d) "plDosage(g.PL)" else "g.GT.nNonRefAlleles()",
          Array("sa.cov.Cov1", "sa.cov.Cov2"))
        .annotateVariantsVDS(
          inputVDS.linreg(Array("sa.pheno", "sa.pheno"),
            if (d) "plDosage(g.PL)" else "g.GT.nNonRefAlleles()",
            Array("sa.cov.Cov1", "sa.cov.Cov2"))
            .annotateVariantsExpr(
              s"""
                 |va.linreg.ytx = [va.linreg.ytx[$i]],
                 |va.linreg.beta = [va.linreg.beta[$i]],
                 |va.linreg.se = [va.linreg.se[$i]],
                 |va.linreg.tstat = [va.linreg.tstat[$i]],
                 |va.linreg.pval = [va.linreg.pval[$i]]
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
}

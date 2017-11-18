package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.{Annotation, Querier}
import is.hail.expr.{TBoolean, TFloat64}
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.Variant
import org.testng.annotations.Test

class LogisticRegressionSuite extends SparkSuite {

  def assertInt(a: Annotation, value: Int) {
    assert(D_==(a.asInstanceOf[Int], value))
  }

  def assertDouble(a: Annotation, value: Double, tol: Double = 1e-6) {
    assert(D_==(a.asInstanceOf[Double], value, tol))
  }

  def assertConsistentWithConstant(converged: Annotation, pval: Annotation) {
    assert(!converged.asInstanceOf[Boolean] || pval.asInstanceOf[Double].isNaN || 1 - pval.asInstanceOf[Double] < 1e-4)
  }

  // x = (0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
  val v1 = Variant("1", 1, "C", "T")
  // x = (., 2, ., 2, 0, 0, 0, 0, 0, 0)
  val v2 = Variant("1", 2, "C", "T")
  // x = (0, ., 1, 1, 1, ., 0, 0, 0, 0)
  val v3 = Variant("1", 3, "C", "T")
  // x = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
  val v6 = Variant("1", 6, "C", "T")
  // x = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
  val v7 = Variant("1", 7, "C", "T")
  // x = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
  val v8 = Variant("1", 8, "C", "T")
  // x = (., 1, 1, 1, 1, 1, 1, 1, 1, 1)
  val v9 = Variant("1", 9, "C", "T")
  // x = (., 2, 2, 2, 2, 2, 2, 2, 2, 2)
  val v10 = Variant("1", 10, "C", "T")

  @Test def waldTestWithTwoCov() {
    val covariates = hc.importTable("src/test/resources/regressionLogistic.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLogisticBoolean.pheno",
      types = Map("isCase" -> TBoolean()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLogistic.vcf")
      .verifyBiallelic()
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .logreg("wald", "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.logreg.beta")._2
    val qSe = vds.queryVA("va.logreg.se")._2
    val qZstat = vds.queryVA("va.logreg.zstat")._2
    val qPVal = vds.queryVA("va.logreg.pval")._2
    val qConverged = vds.queryVA("va.logreg.fit.converged")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)

    logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    waldtest <- coef(summary(logfit))
    beta <- waldtest["x", "Estimate"]
    se <- waldtest["x", "Std. Error"]
    zstat <- waldtest["x", "z value"]
    pval <- waldtest["x", "Pr(>|z|)"]
    */

    assertDouble(qBeta(a(v1)), -0.81226793796)
    assertDouble(qSe(a(v1)), 2.1085483421)
    assertDouble(qZstat(a(v1)), -0.3852261396)
    assertDouble(qPVal(a(v1)), 0.7000698784)

    // v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(.5, 2, .5, 2, 0, 0, 0, 0, 0, 0)

    assertDouble(qBeta(a(v2)), -0.43659460858)
    assertDouble(qSe(a(v2)), 1.0296902941)
    assertDouble(qZstat(a(v2)), -0.4240057531)
    assertDouble(qPVal(a(v2)), 0.6715616176)

    // seperable => does not converge
    // v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0, 0.375, 1, 1, 1, 0.375, 0, 0, 0, 0)

    // val qExploded = vds.queryVA("va.logreg.fit.exploded")._2
    // assertExploded(v3), explodes at iteration ~30, but we cap at 25
    assert(!qConverged(a(v3)).asInstanceOf[Boolean])

    // constant genotypes after imputation
    assert(!qConverged(a(v6)).asInstanceOf[Boolean])
    assert(!qConverged(a(v7)).asInstanceOf[Boolean])
    assert(!qConverged(a(v8)).asInstanceOf[Boolean])
    assert(!qConverged(a(v9)).asInstanceOf[Boolean])
    assert(!qConverged(a(v10)).asInstanceOf[Boolean])
  }

  @Test def waldTestWithTwoCovPhred() {
    val covariates = hc.importTable("src/test/resources/regressionLogistic.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLogisticBoolean.pheno",
      types = Map("isCase" -> TBoolean()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLogistic.vcf")
      .verifyBiallelic()
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .logreg("wald", "sa.pheno", "plDosage(g.PL)", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.logreg.beta")._2
    val qSe = vds.queryVA("va.logreg.se")._2
    val qZstat = vds.queryVA("va.logreg.zstat")._2
    val qPVal = vds.queryVA("va.logreg.pval")._2
    val qConverged = vds.queryVA("va.logreg.fit.converged")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    x = c(0.009900990296049406, 0.9900990100009803, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.9900990100009803, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406)
    y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)

    logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    waldtest <- coef(summary(logfit))
    beta <- waldtest["x", "Estimate"]
    se <- waldtest["x", "Std. Error"]
    zstat <- waldtest["x", "z value"]
    pval <- waldtest["x", "Pr(>|z|)"]
    */

    assertDouble(qBeta(a(v1)), -0.8286774)
    assertDouble(qSe(a(v1)), 2.151145)
    assertDouble(qZstat(a(v1)), -0.3852261)
    assertDouble(qPVal(a(v1)), 0.7000699)

    // v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0.50247524764, 1.980198019704931, 0.50247524764, 1.980198019704931, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406)

    assertDouble(qBeta(a(v2)), -0.4431764)
    assertDouble(qSe(a(v2)), 1.045213)
    assertDouble(qZstat(a(v2)), -0.4240058)
    assertDouble(qPVal(a(v2)), 0.6715616)

    // seperable => does not converge
    // v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0, 0.375, 1, 1, 1, 0.375, 0, 0, 0, 0)
    assert(!qConverged(a(v3)).asInstanceOf[Boolean])

    // constant dosage, numerically unstable
    assertConsistentWithConstant(qConverged(a(v6)), qPVal(a(v6)))
    assertConsistentWithConstant(qConverged(a(v7)), qPVal(a(v7)))
    assertConsistentWithConstant(qConverged(a(v8)), qPVal(a(v8)))
    assertConsistentWithConstant(qConverged(a(v9)), qPVal(a(v9)))
    assertConsistentWithConstant(qConverged(a(v10)), qPVal(a(v10)))
  }

  @Test def waldTestWithTwoCovDosage() {
    val covariates = hc.importTable("src/test/resources/regressionLogistic.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLogisticBoolean.pheno",
      types = Map("isCase" -> TBoolean()), missing = "0").keyBy("Sample")

    // .gen and .sample files created from regressionLogistic.vcf
    // dosages are derived from PLs so results should agree with testWithTwoCovPhred
    val vds =  hc.importGen("src/test/resources/regressionLogistic.gen", "src/test/resources/regressionLogistic.sample")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .logreg("wald", "sa.pheno", "dosage(g.GP)", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qBeta = vds.queryVA("va.logreg.beta")._2
    val qSe = vds.queryVA("va.logreg.se")._2
    val qZstat = vds.queryVA("va.logreg.zstat")._2
    val qPVal = vds.queryVA("va.logreg.pval")._2
    val qConverged = vds.queryVA("va.logreg.fit.converged")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    x = c(0.009900990296049406, 0.9900990100009803, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.9900990100009803, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406)
    y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)

    logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    waldtest <- coef(summary(logfit))
    beta <- waldtest["x", "Estimate"]
    se <- waldtest["x", "Std. Error"]
    zstat <- waldtest["x", "z value"]
    pval <- waldtest["x", "Pr(>|z|)"]
    */

    assertDouble(qBeta(a(v1)), -0.8286774, 1e-4)
    assertDouble(qSe(a(v1)), 2.151145, 1e-4)
    assertDouble(qZstat(a(v1)), -0.3852261, 1e-4)
    assertDouble(qPVal(a(v1)), 0.7000699, 1e-4)

    // v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0.50247524764, 1.980198019704931, 0.50247524764, 1.980198019704931, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406, 0.009900990296049406)

    assertDouble(qBeta(a(v2)), -0.4431764, 1e-4)
    assertDouble(qSe(a(v2)), 1.045213, 1e-4)
    assertDouble(qZstat(a(v2)), -0.4240058, 1e-4)
    assertDouble(qPVal(a(v2)), 0.6715616, 1e-4)

    // seperable => does not converge
    // v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0, 0.375, 1, 1, 1, 0.375, 0, 0, 0, 0)
    assert(!qConverged(a(v3)).asInstanceOf[Boolean])

    // constant dosage, numerically unstable
    assertConsistentWithConstant(qConverged(a(v6)), qPVal(a(v6)))
    assertConsistentWithConstant(qConverged(a(v7)), qPVal(a(v7)))
    assertConsistentWithConstant(qConverged(a(v8)), qPVal(a(v8)))
    assertConsistentWithConstant(qConverged(a(v9)), qPVal(a(v9)))
    assertConsistentWithConstant(qConverged(a(v10)), qPVal(a(v10)))
  }

  @Test def lrTestWithTwoCov() {
    val covariates = hc.importTable("src/test/resources/regressionLogistic.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLogisticBoolean.pheno",
      types = Map("isCase" -> TBoolean()), missing = "0").keyBy("Sample")


    val vds = hc.importVCF("src/test/resources/regressionLogistic.vcf")
      .verifyBiallelic()
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .logreg("lrt", "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))


    val qBeta = vds.queryVA("va.logreg.beta")._2
    val qChi2 = vds.queryVA("va.logreg.chi2")._2
    val qPVal = vds.queryVA("va.logreg.pval")._2
    val qConverged = vds.queryVA("va.logreg.fit.converged")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)

    logfit <- glm(y ~ x + c1 + c2, family=binomial(link="logit"))
    logfitnull <- glm(y ~ c1 + c2, family=binomial(link="logit"))
    beta <- coef(summary(logfit))["x", "Estimate"]
    lrtest <- anova(logfitnull, logfit, test="LRT")
    chi2 <- lrtest[["Deviance"]][2]
    pval <- lrtest[["Pr(>Chi)"]][2]
    */

    assertDouble(qBeta(a(v1)), -0.81226793796)
    assertDouble(qChi2(a(v1)), 0.1503349167)
    assertDouble(qPVal(a(v1)), 0.6982155052)

    // v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(.5, 2, .5, 2, 0, 0, 0, 0, 0, 0)

    assertDouble(qBeta(a(v2)), -0.43659460858)
    assertDouble(qChi2(a(v2)), 0.1813968574)
    assertDouble(qPVal(a(v2)), 0.6701755415)

    // seperable => does not converge
    // v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0, 0.375, 1, 1, 1, 0.375, 0, 0, 0, 0)

    // val qExploded = vds.queryVA("va.logreg.fit.exploded")._2
    // assertExploded(v3), explodes at iteration ~30, but we cap at 25
    assert(!qConverged(a(v3)).asInstanceOf[Boolean])

    // constant genotypes after imputation
    assert(!qConverged(a(v6)).asInstanceOf[Boolean])
    assert(!qConverged(a(v7)).asInstanceOf[Boolean])
    assert(!qConverged(a(v8)).asInstanceOf[Boolean])
    assert(!qConverged(a(v9)).asInstanceOf[Boolean])
    assert(!qConverged(a(v10)).asInstanceOf[Boolean])
  }

  @Test def scoreTestWithTwoCov() {
    val covariates = hc.importTable("src/test/resources/regressionLogistic.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLogisticBoolean.pheno",
      types = Map("isCase" -> TBoolean()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLogistic.vcf")
      .verifyBiallelic()
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")
      .logreg("score", "sa.pheno", "g.GT.nNonRefAlleles()", Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val qChi2 = vds.queryVA("va.logreg.chi2")._2
    val qPVal = vds.queryVA("va.logreg.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    /*
    comparing to output of R code:
    x = c(0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
    y = c(0, 0, 1, 1, 1, 1, 0, 0, 1, 1)
    c1 = c(0, 2, 1, -2, -2, 4, 1, 2, 3, 4)
    c2 = c(-1, 3, 5, 0, -4, 3, 0, -2, -1, -4)

    logfit <- glm(y ~ c1 + c2 + x, family=binomial(link="logit"))
    logfitnull <- glm(y ~ c1 + c2, family=binomial(link="logit"))
    scoretest <- anova(logfitnull, logfit, test="Rao")
    chi2 <- scoretest[["Rao"]][2]
    pval <- scoretest[["Pr(>Chi)"]][2]
    */

    assertDouble(qChi2(a(v1)), 0.1502364955)
    assertDouble(qPVal(a(v1)), 0.6983094571)

    // v2 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(.5, 2, .5, 2, 0, 0, 0, 0, 0, 0)

    assertDouble(qChi2(a(v2)), 0.1823600965)
    assertDouble(qPVal(a(v2)), 0.6693528073)

    // v3 has two missing genotypes, comparing to output of R code as above with imputed genotypes:
    // x = c(0, 0.375, 1, 1, 1, 0.375, 0, 0, 0, 0)

    assertDouble(qChi2(a(v3)), 7.047367694)
    assertDouble(qPVal(a(v3)), 0.007938182229)

    // constant genotypes after imputation
    assert(qChi2(a(v6)) == null || (qChi2(a(v6)).asInstanceOf[Double] < 1e-6))
    assert(qChi2(a(v7)) == null || (qChi2(a(v7)).asInstanceOf[Double] < 1e-6))
    assert(qChi2(a(v8)) == null || (qChi2(a(v8)).asInstanceOf[Double] < 1e-6))
    assert(qChi2(a(v9)) == null || (qChi2(a(v9)).asInstanceOf[Double] < 1e-6))
    assert(qChi2(a(v10)) == null || (qChi2(a(v10)).asInstanceOf[Double] < 1e-6))
  }

  @Test def waldEpactsTest() {

    val covariates = hc.importTable("src/test/resources/regressionLogisticEpacts.cov",
      types = Map("PC1" -> TFloat64(), "PC2" -> TFloat64()), missing = "0").keyBy("IND_ID")

    val vds = hc.importVCF("src/test/resources/regressionLogisticEpacts.vcf")
      .verifyBiallelic()
      .annotateSamplesTable(KeyTable.importFam(hc, "src/test/resources/regressionLogisticEpacts.fam"), root = "sa.fam")
      .annotateSamplesTable(covariates, root = "sa.pc")
      .logreg("wald", "sa.fam.isCase", "g.GT.nNonRefAlleles()", Array("sa.fam.isFemale", "sa.pc.PC1", "sa.pc.PC2"), "va.wald")
      .logreg("lrt", "sa.fam.isCase", "g.GT.nNonRefAlleles()", Array("sa.fam.isFemale", "sa.pc.PC1", "sa.pc.PC2"), "va.lrt")
      .logreg("score", "sa.fam.isCase", "g.GT.nNonRefAlleles()", Array("sa.fam.isFemale", "sa.pc.PC1", "sa.pc.PC2"), "va.score")
      .logreg("firth", "sa.fam.isCase", "g.GT.nNonRefAlleles()", Array("sa.fam.isFemale", "sa.pc.PC1", "sa.pc.PC2"), "va.firth")

    // 2535 samples from 1K Genomes Project
    val v1 = Variant("22", 16060511, "T", "TTC")
    // MAC  623
    val v2 = Variant("22", 16115878, "G", "C")
    // MAC  370
    val v3 = Variant("22", 16115882, "G", "T")
    // MAC 1207
    val v4 = Variant("22", 16117940, "T", "G")
    // MAC    7
    val v5 = Variant("22", 16117953, "G", "C") // MAC   21

    val qBeta = vds.queryVA("va.wald.beta")._2
    val qSe = vds.queryVA("va.wald.se")._2
    val qZstat = vds.queryVA("va.wald.zstat")._2
    val qPVal = vds.queryVA("va.wald.pval")._2
    val qPValLR = vds.queryVA("va.lrt.pval")._2
    val qPValScore = vds.queryVA("va.score.pval")._2
    val qBetaFirth = vds.queryVA("va.firth.beta")._2
    val qPValFirth = vds.queryVA("va.firth.pval")._2

    val a = vds.variantsAndAnnotations.collect().toMap

    // Comparing to output of b.wald, b.lrt, and b.score in EPACTS
    // for five 1KG project variants with no missing genotypes
    // http://genome.sph.umich.edu/wiki/EPACTS#Single_Variant_Tests

    assertDouble(qBeta(a(v1)), -0.097476, 1e-4)
    assertDouble(qSe(a(v1)), 0.087478, 1e-4)
    assertDouble(qZstat(a(v1)), -1.1143, 1e-3)
    assertDouble(qPVal(a(v1)), 0.26516, 1e-4)
    assertDouble(qPValLR(a(v1)), 0.26475, 1e-4)
    assertDouble(qPValScore(a(v1)), 0.26499, 1e-4)
    assertDouble(qBetaFirth(a(v1)), -0.097079, 1e-4)
    assertDouble(qPValFirth(a(v1)), 0.26593, 1e-4)

    assertDouble(qBeta(a(v2)), -0.052632, 1e-4)
    assertDouble(qSe(a(v2)), 0.11272, 1e-4)
    assertDouble(qZstat(a(v2)), -0.46691, 1e-4)
    assertDouble(qPVal(a(v2)), 0.64056, 1e-4)
    assertDouble(qPValLR(a(v2)), 0.64046, 1e-4)
    assertDouble(qPValScore(a(v2)), 0.64054, 1e-4)
    assertDouble(qBetaFirth(a(v2)), -0.052301, 1e-4)
    assertDouble(qPValFirth(a(v2)), 0.64197, 1e-4)

    assertDouble(qBeta(a(v3)), -0.15598, 1e-4)
    assertDouble(qSe(a(v3)), 0.079508, 1e-4)
    assertDouble(qZstat(a(v3)), -1.9619, 1e-4)
    assertDouble(qPVal(a(v3)), 0.049779, 1e-4)
    assertDouble(qPValLR(a(v3)), 0.049675, 1e-4)
    assertDouble(qPValScore(a(v3)), 0.049717, 1e-4)
    assertDouble(qBetaFirth(a(v3)), -0.15567, 1e-4)
    assertDouble(qPValFirth(a(v3)), 0.04991, 1e-4)

    assertDouble(qBeta(a(v4)), -0.88059, 1e-4)
    assertDouble(qSe(a(v4)), 0.83769, 1e-3)
    assertDouble(qZstat(a(v4)), -1.0512, 1e-3)
    assertDouble(qPVal(a(v4)), 0.29316, 1e-2)
    assertDouble(qPValLR(a(v4)), 0.26984, 1e-4)
    assertDouble(qPValScore(a(v4)), 0.27828, 1e-4)
    assertDouble(qBetaFirth(a(v4)), -0.7524, 1e-4)
    assertDouble(qPValFirth(a(v4)), 0.30731, 1e-4)

    assertDouble(qBeta(a(v5)), 0.54921, 1e-4)
    assertDouble(qSe(a(v5)), 0.4517, 1e-3)
    assertDouble(qZstat(a(v5)), 1.2159, 1e-3)
    assertDouble(qPVal(a(v5)), 0.22403, 1e-3)
    assertDouble(qPValLR(a(v5)), 0.21692, 1e-4)
    assertDouble(qPValScore(a(v5)), 0.21849, 1e-4)
    assertDouble(qBetaFirth(a(v5)), 0.5258, 1e-4)
    assertDouble(qPValFirth(a(v5)), 0.22562, 1e-4)
  }
}

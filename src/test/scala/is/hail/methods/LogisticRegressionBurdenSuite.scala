package is.hail.methods

import is.hail.SparkSuite
import is.hail.TestUtils.interceptFatal
import is.hail.expr.TDouble
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.variant.VariantDataset
import org.testng.annotations.Test

class LogisticRegressionBurdenSuite extends SparkSuite {
  /*
  Three genes overlap each other in the first 3 positions:

              Position
        1 2 3 4 5 6 7 8 9 10
  Gene1 ---
  Gene2   -
  Gene3 -----
  */

  val vdsBurden: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.genes", all=true)
    .annotateVariantsExpr("va.weight = v.start.toDouble")
    .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
      "Sample",
      root = Some("sa.pheno0"),
      config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
    .annotateSamplesExpr("sa.pheno.Pheno = if (sa.pheno0.Pheno == 1.0) false else if (sa.pheno0.Pheno == 2.0) true else NA: Boolean")
    .annotateSamplesTable("src/test/resources/regressionLinear.cov",
      "Sample",
      root = Some("sa.cov"),
      config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))

  @Test def testMax() {
    val (logregWaldKT, sampleKT) = vdsBurden.logregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => g.gt.toDouble).max()", "wald", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    val (logregLRTKT, _) = vdsBurden.logregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => g.gt.toDouble).max()", "lrt", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    val (logregScoreKT, _) = vdsBurden.logregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => g.gt.toDouble).max()", "score", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val sampleMap = keyTableBoxedDoubleToMap[String](sampleKT)

    val logregWaldMap = keyTableBoxedDoubleToMap[String](logregWaldKT.select(logregWaldKT.fieldNames.dropRight(1), logregWaldKT.key))
    val logregLRTMap = keyTableBoxedDoubleToMap[String](logregLRTKT.select(logregLRTKT.fieldNames.dropRight(1), logregLRTKT.key))
    val logregScoreMap = keyTableBoxedDoubleToMap[String](logregScoreKT)

    /*
    Variant A       B       C       D       E       F
    1       0.0     1.0     0.0     0.0     0.0     1.0
    2         .     2.0       .     2.0     0.0     0.0
    3       0.0       .     1.0     1.0     1.0       .

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0    -1.0     0.0     0.0       .     0.0
    B       2.0     3.0     0.0     2.0     2.0     2.0
    C       1.0     5.0     1.0     0.0       .     1.0
    D      -2.0     0.0     1.0     2.0     2.0     2.0
    E      -2.0    -4.0     1.0     0.0     0.0     1.0
    F       4.0     3.0     1.0     1.0     0.0     1.0
    */

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(0.0, 2.0, 0.0, 2.0, 0.0, 1.0),
      2 -> IndexedSeq(null, 2.0, null, 2.0, 0.0, 0.0),
      3 -> IndexedSeq(0.0, 2.0, 1.0, 2.0, 1.0, 1.0))

    // Values from R: logfit <- glm(Pheno ~ Gene1 + Cov1 + Cov2, data=df, family=binomial(link="logit"))
    val logregWaldMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.4618204, 1.171908, -0.3940755, 0.6935253),
      2 -> IndexedSeq(null, null, null, null),
      3 -> IndexedSeq(0.3680554, 1.437571, 0.2560259, 0.7979308))

    // Values from R:
    // logfit <- glm(Pheno ~ Gene1 + Cov1 + Cov2, data=df, family=binomial(link="logit"))
    // logfitnull <- glm(Pheno ~ Cov1 + Cov2, data=df, family=binomial(link="logit"))
    // beta <- coef(summary(logfit))["x", "Estimate"]
    // lrtest <- anova(logfitnull, logfit, test="LRT")
    val logregLRTMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.4618204, 0.166406, 0.6833259),
      2 -> IndexedSeq(null, null, null, null),
      3 -> IndexedSeq(0.3680554, 0.06593895, 0.7973441))

    // Values from R:
    // logfit <- glm(Pheno ~ Gene1 + Cov1 + Cov2, data=df, family=binomial(link="logit"))
    // logfitnull <- glm(Pheno ~ Cov1 + Cov2, data=df, family=binomial(link="logit"))
    // scoretest <- anova(logfitnull, logfit, test="Rao")
    val logregScoreMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(0.163201, 0.6862264),
      2 -> IndexedSeq(3.50656, 0.06112625),
      3 -> IndexedSeq(0.06615483, 0.7970199))

    assert(mapSameElements(sampleMapR.map{ case (k, v) => (s"Gene$k", v) }, sampleMap, indexedSeqBoxedDoubleEquals(1e-6)))

    assert(mapSameElements(logregWaldMapR.map{ case (k, v) => (s"Gene$k", v) },
      logregWaldMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(logregLRTMapR.map{ case (k, v) => (s"Gene$k", v) },
      logregLRTMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(logregScoreMapR.map{ case (k, v) => (s"Gene$k", v) },
      logregScoreMap, indexedSeqBoxedDoubleEquals(1e-3)))
  }

  @Test def testFatals() {
    interceptFatal("clashes with reserved wald logreg columns") {
      vdsBurden.logregBurden("pval", "va.genes", singleKey = false,
        "gs.map(g => g.gt.toDouble).max()", "wald", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }

    interceptFatal("clashes with a sample name") {
      vdsBurden.logregBurden("A", "va.genes", singleKey = false,
        "gs.map(g => g.gt.toDouble).max()", "wald", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }

    interceptFatal("phenotype must be Boolean or numeric with all present values equal to 0 or 1") {
      vdsBurden.logregBurden("gene", "va.genes", singleKey = false,
        "gs.map(g => g.gt.toDouble).max()", "wald", "sa.pheno0.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }
  }
}

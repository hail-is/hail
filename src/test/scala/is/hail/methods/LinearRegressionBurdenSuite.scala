package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.{TDouble, TString}
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.io.annotators.IntervalList
import is.hail.variant.VariantDataset
import org.testng.annotations.Test

class LinearRegressionBurdenSuite extends SparkSuite {
  /*
  vdsBurden is shared by testWeightedSum, testWeightedSumWithImputation, and testMax.
  Three genes overlap each other in the first 3 positions:

              Position
        1 2 3 4 5 6 7 8 9 10
  Gene1 ---
  Gene2   -
  Gene3 -----

  Here are the genotypes for the six complete samples:

  Variant A       B       C       D       E       F
  1       0.0     1.0     0.0     0.0     0.0     1.0
  2         .     2.0       .     2.0     0.0     0.0
  3       0.0       .     1.0     1.0     1.0       .
  */

  def intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")
  def covariates = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)).keyBy("Sample")
  def phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TDouble), missing = "0").keyBy("Sample")

  lazy val vdsBurden: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsTable(intervals, root="va.genes", product=true)
    .annotateVariantsExpr("va.weight = v.start.toDouble")
    .annotateSamplesTable(covariates, root = "sa.cov")
    .annotateSamplesTable(phenotypes, root = "sa.pheno")

  @Test def testWeightedSum() {

    val (linregKT, sampleKT) = vdsBurden.linregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val linregMap = keyTableBoxedDoubleToMap[String](linregKT)
    val sampleMap = keyTableBoxedDoubleToMap[String](sampleKT)

    /*
    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0    -1.0     1.0     0.0     0.0     0.0
    B       2.0     3.0     1.0     5.0     4.0     5.0
    C       1.0     5.0     2.0     0.0     0.0     3.0
    D      -2.0     0.0     2.0     4.0     4.0     7.0
    E      -2.0    -4.0     2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     1.0
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.08164, 0.15339, -0.532, 0.6478),
      2 -> IndexedSeq(-0.09900, 0.17211, -0.575, 0.6233),
      3 -> IndexedSeq(0.01558, 0.18323, 0.085, 0.940))

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(0.0, 5.0, 0.0, 4.0, 0.0, 1.0),
      2 -> IndexedSeq(0.0, 4.0, 0.0, 4.0, 0.0, 0.0),
      3 -> IndexedSeq(0.0, 5.0, 3.0, 7.0, 3.0, 1.0))

    assert(mapSameElements(linregMapR.map{ case (k, v) => (s"Gene$k", v) }, linregMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(sampleMapR.map{ case (k, v) => (s"Gene$k", v) }, sampleMap, indexedSeqBoxedDoubleEquals(1e-6)))
  }

  @Test def testWeightedSumWithImputation() {

    val (linregKT, sampleKT) = vdsBurden
      .filterSamplesExpr("isDefined(sa.pheno) && isDefined(sa.cov.Cov1) && isDefined(sa.cov.Cov2)")
      .variantQC()
      .linregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => va.weight * orElse(g.gt.toDouble, 2 * va.qc.AF)).sum()",
      "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val linregMap = keyTableBoxedDoubleToMap[String](linregKT)
    val sampleMap = keyTableBoxedDoubleToMap[String](sampleKT)

    /*
    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0    -1.0     1.0     2.0     2.0     2.0
    B       2.0     3.0     1.0     5.0     4.0     7.25
    C       1.0     5.0     2.0     2.0     2.0     5.0
    D      -2.0     0.0     2.0     4.0     4.0     7.0
    E      -2.0    -4.0     2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     3.25
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.2177, 0.1621, -1.343, 0.3115),
      2 -> IndexedSeq(-0.2709, 0.1675, -1.617, 0.2473),
      3 -> IndexedSeq(-0.05710, 0.21078, -0.271, 0.812))

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(2.0, 5.0, 2.0, 4.0, 0.0, 1.0),
      2 -> IndexedSeq(2.0, 4.0, 2.0, 4.0, 0.0, 0.0),
      3 -> IndexedSeq(2.0, 7.25, 5.0, 7.0, 3.0, 3.25))

    assert(mapSameElements(linregMapR.map{ case (k, v) => (s"Gene$k", v) }, linregMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(sampleMapR.map{ case (k, v) => (s"Gene$k", v) }, sampleMap, indexedSeqBoxedDoubleEquals(1e-6)))
    }

  @Test def testMax() {
    val (linregKT, sampleKT) = vdsBurden.linregBurden("gene", "va.genes", singleKey = false,
      "gs.map(g => g.gt.toDouble).max()", "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val linregMap = keyTableBoxedDoubleToMap[String](linregKT)
    val sampleMap = keyTableBoxedDoubleToMap[String](sampleKT)

    /*
    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0    -1.0     1.0     0.0       .     0.0
    B       2.0     3.0     1.0     2.0     2.0     2.0
    C       1.0     5.0     2.0     0.0       .     1.0
    D      -2.0     0.0     2.0     2.0     2.0     2.0
    E      -2.0    -4.0     2.0     0.0     0.0     1.0
    F       4.0     3.0     2.0     1.0     0.0     1.0
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.08368, 0.36841, -0.227, 0.8414),
      2 -> IndexedSeq(-0.5418, 0.3351, -1.617, 0.2473),
      3 -> IndexedSeq(0.07474, 0.51528, 0.145, 0.898))

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(0.0, 2.0, 0.0, 2.0, 0.0, 1.0),
      2 -> IndexedSeq(null, 2.0, null, 2.0, 0.0, 0.0),
      3 -> IndexedSeq(0.0, 2.0, 1.0, 2.0, 1.0, 1.0))


    assert(mapSameElements(linregMapR.map{ case (k, v) => (s"Gene$k", v) }, linregMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(sampleMapR.map{ case (k, v) => (s"Gene$k", v) }, sampleMap, indexedSeqBoxedDoubleEquals(1e-6)))
  }

  @Test def testSingleVsArray() {
    /*
    Three disjoint genes are in the first three positions.

                Position
          1 2 3 4 5 6 7 8 9 10
    Gene1 -
    Gene2   -
    Gene3     -
    */

    val vdsBurdenNoOverlap: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateVariantsTable(IntervalList.read(hc, "src/test/resources/regressionLinearNoOverlap.interval_list"), root="va.gene")
      .annotateVariantsExpr("va.weight = v.start.toDouble")
      .annotateVariantsExpr("""va.genes2 = if (isDefined(va.gene)) [va.gene] else range(0).map(x => "")""")
      .annotateVariantsExpr("va.genes3 = if (isDefined(va.gene)) [va.gene] else NA: Array[String]")
      .annotateVariantsExpr("va.genes4 = if (isDefined(va.gene)) [va.gene, va.gene].toSet else NA: Set[String]")
      .annotateVariantsExpr("va.genes5 = if (isDefined(va.gene)) [va.gene, va.gene] else NA: Array[String]")
      .annotateVariantsExpr("va.genes6 = NA: Set[String]")
      .annotateVariantsExpr("va.genes7 = v.start")
      .annotateVariantsExpr("va.genes8 = v.start.toDouble")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")

    val (linregKT, sampleKT) = vdsBurdenNoOverlap.linregBurden("gene", "va.gene", singleKey = true,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1 + 1 - 1"))

    val (linregKT2, sampleKT2) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes2", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT3, sampleKT3) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes3", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT4, sampleKT4) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes4", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT5, sampleKT5) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes5", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT6, sampleKT6) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes6", singleKey = false,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT7, sampleKT7) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes7", singleKey = true,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT8, sampleKT8) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes8", singleKey = true,
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno", covariates = Array("sa.cov.Cov1"))

    assert(linregKT same linregKT2)
    assert(sampleKT same sampleKT2)

    assert(linregKT same linregKT3)
    assert(sampleKT same sampleKT3)

    assert(linregKT same linregKT4)
    assert(sampleKT same sampleKT4)

    val twiceSampleMap = keyTableBoxedDoubleToMap[String](sampleKT).mapValues(_.map(v =>
      if (v == null) null else box(2 * v.doubleValue())))
    val onceSampleMap5 = keyTableBoxedDoubleToMap[String](sampleKT5)
    assert(twiceSampleMap == onceSampleMap5)

    assert(linregKT6.count == 0)
    assert(sampleKT6.count == 0)

    val sampleMap7 = keyTableBoxedDoubleToMap[Int](sampleKT7)
    assert(sampleMap7.size == 10)
    assert(sampleMap7.forall { case (key, value) => value.forall(_ % key.asInstanceOf[Int] == 0) })

    val sampleMap8 = keyTableBoxedDoubleToMap[Double](sampleKT8)
    assert(sampleMap8.size == 10)
    assert(sampleMap8.forall { case (key, value) => value.forall(_ % key.asInstanceOf[Double] == 0) })
  }

  @Test def testFatals() {
    interceptFatal("clashes with reserved linreg columns") {
      vdsBurden.linregBurden("pval", "va.genes", singleKey = false,
        "gs.map(g => g.gt.toDouble).max()", "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }

    interceptFatal("clashes with a sample name") {
      vdsBurden.linregBurden("A", "va.genes", singleKey = false,
        "gs.map(g => g.gt.toDouble).max()", "sa.pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    }
  }
}

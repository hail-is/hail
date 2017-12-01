package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.expr.{TFloat64, TInt64}
import is.hail.io.annotators.IntervalList
import org.testng.annotations.Test
import is.hail.TestUtils._
import is.hail.check.Prop.forAll
import is.hail.utils._
import is.hail.variant.{VSMSubgen, VariantSampleMatrix}


class GroupBySuite extends SparkSuite {

  @Test def testGroupVariantsBy() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateVariantsExpr("va.foo = gs.filter(g => g.isDefined).map(g => g.gt).sum()")
    val vds2 = vds.groupVariantsBy("va.foo", "gs.map(g => g.gt).max()", true)
  }

  @Test def testRandomVSMEquivalence() {
    var skipped = 0
    val p = forAll(VariantSampleMatrix.gen(hc, VSMSubgen.random)) { vsm =>
      val variants = vsm.variants.collect()
      val uniqueVariants = variants.toSet
      if (variants.length != uniqueVariants.size) {
        skipped += 1
        val grouped = vsm.groupVariantsBy("v", "gs.collect()[0]", true)
        grouped.countVariants() == uniqueVariants.size
      } else {
        val vaKT = vsm.variantsKT()
        val grouped = vsm.groupVariantsBy("v", "gs.collect()[0]", true)
        vsm.annotateVariantsExpr("va = {}").same(grouped)
      }
    }
    p.check()
    if (skipped != 0)
    println(s"warning: skipped $skipped evaluations due to non-unique variants.")
  }

  @Test def testLinregBurden() {
    val intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateVariantsTable(intervals, root="va.genes", product=true)
      .annotateVariantsExpr("va.weight = v.start.toFloat64")
      .annotateSamplesTable(covariates, root = "sa.cov")
      .annotateSamplesTable(phenotypes, root = "sa.pheno")

    val vdsGrouped = vds.groupVariantsBy("va.genes", "gs.map(g => va.weight * g.gt).sum()")
    println(vds.rowType)
    println(vdsGrouped.rowType)

    val resultsVSM = vdsGrouped.linreg(Array("sa.pheno"), "g", covExpr = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    val linregMap = resultsVSM.variantsKT().select("v", "va.linreg.beta", "va.linreg.se", "va.linreg.tstat", "va.linreg.pval")
      .mapAnnotations { r => (r.getAs[String](0), (1 to 4).map{ i => Double.box(r.getAs[IndexedSeq[Double]](i)(0)) }) }
      .collect()
      .toMap
    val sampleMap = resultsVSM.unsafeRowRDD().map {ur =>
      val k = ur.getAs[String](0)
      val v = ur.getAs[IndexedSeq[Double]](3)
      k -> v.map(Double.box)
    }.collect().toMap

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
      1 -> IndexedSeq(0.0, 5.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0),
      2 -> IndexedSeq(0.0, 4.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
      3 -> IndexedSeq(0.0, 5.0, 3.0, 0.0, 7.0, 3.0, 1.0, 0.0))

    assert(mapSameElements(linregMapR.map { case (k, v) => (s"Gene$k", v) }, linregMap, indexedSeqBoxedDoubleEquals(1e-3)))
    assert(mapSameElements(sampleMapR.map{ case (k, v) => (s"Gene$k", v) }, sampleMap, indexedSeqBoxedDoubleEquals(1e-6)))
  }
}

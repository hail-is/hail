package is.hail.variant.vsm

import is.hail.SparkSuite
import is.hail.io.annotators.IntervalList
import org.testng.annotations.Test
import is.hail.TestUtils._
import is.hail.annotations.{Annotation, UnsafeRow}
import is.hail.check.Prop.forAll
import is.hail.expr.types._
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant.{MatrixTable, VSMSubgen}
import org.apache.spark.sql.Row

class GroupBySuite extends SparkSuite {

  @Test def testGroupSamplesBy() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateSamplesExpr("AC = gs.map(g => g.GT.gt).sum()")
    val vds2 = vds.groupSamplesBy("AC = sa.AC", "max = gs.map(g => g.GT.gt).max()").count()
  }

  @Test def testGroupSamplesStruct() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateSamplesExpr("foo = {str1: 1, str2: \"bar\"}")
    val vds2 = vds.groupSamplesBy("foo = sa.foo", "max = gs.map(g => g.GT.gt).max()").count()
  }

  @Test def testGroupVariantsBy() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateVariantsExpr("AC = gs.map(g => g.GT.gt).sum()")
    val vds2 = vds.groupVariantsBy("AC = va.AC", "max = gs.map(g => g.GT.gt).max()").count()
  }

  @Test def testGroupVariantsStruct() {
    val vds = hc.importVCF("src/test/resources/sample.vcf").annotateVariantsExpr("AC = {str1: \"foo\", str2: 1}")
    val vds2 = vds.groupVariantsBy("AC = va.AC", "max = gs.map(g => g.GT.gt).max()").count()
  }

  // FIXME this test is broken, @tpoterba has fixed in a separate branch
  @Test(enabled = false) def testRandomVSMEquivalence() {
    var vSkipped = 0
    var sSkipped = 0
    val p = forAll(MatrixTable.gen(hc, VSMSubgen.random)
      .map(_.unfilterEntries())
    ) { vsm =>
      val variants = vsm.variants.collect()
      val uniqueVariants = variants.toSet
      if (variants.length != uniqueVariants.size) {
        vSkipped += 1
        val grouped = vsm.groupVariantsBy("locus = va.locus, alleles = va.alleles",
          "first = gs.collect()[0]",
          Some(Array("locus")))
        grouped.countVariants() == uniqueVariants.size
      } else {
        val grouped = vsm
          .groupVariantsBy("locus = va.locus, alleles = va.alleles",
            "GT = gs.collect()[0].GT, " +
              "AD = gs.collect()[0].AD, " +
              "DP = gs.collect()[0].DP, " +
              "GQ = gs.collect()[0].GQ, " +
              "PL = gs.collect()[0].PL",
            Some(Array("locus")))
        assert(vsm.selectRows("va.locus", "va.alleles").same(grouped))
      }

      val uniqueSamples = vsm.stringSampleIds.toSet
      if (vsm.stringSampleIds.size != uniqueSamples.size) {
        sSkipped += 1
        val grouped = vsm.groupSamplesBy("s = sa.s", "first = gs.collect()[0]")
        grouped.numCols == uniqueSamples.size
      } else {
        val grouped = vsm.groupSamplesBy("s = sa.s", "GT = gs.collect()[0].GT, AD = gs.collect()[0].AD, DP = gs.collect()[0].DP, GQ = gs.collect()[0].GQ, PL = gs.collect()[0].PL")
        assert(vsm.selectCols("sa.s")
          .same(grouped
            .reorderSamples(vsm.stringSampleIds.toArray.map(Annotation(_)))
          ))
      }
      true
    }
    p.check()
    if (sSkipped != 0)
      println(s"warning: skipped $sSkipped evaluations due to non-unique samples.")
    if (vSkipped != 0)
      println(s"warning: skipped $vSkipped evaluations due to non-unique variants.")
  }

  @Test def testLinregBurden() {
    val intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")
    val covariates = hc.importTable("src/test/resources/regressionLinear.cov",
      types = Map("Cov1" -> TFloat64(), "Cov2" -> TFloat64())).keyBy("Sample")
    val phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
      types = Map("Pheno" -> TFloat64()), missing = "0").keyBy("Sample")

    val vds = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateVariantsTable(intervals, root = "genes", product = true)
      .annotateVariantsExpr("genes = va.genes.map(x => x.target)")
      .annotateVariantsExpr("weight = va.locus.position.toFloat64")
      .annotateSamplesTable(covariates, root = "cov")
      .annotateSamplesTable(phenotypes, root = "pheno")

    val vdsGrouped = vds.explodeVariants("va.genes").groupVariantsBy("genes = va.genes", "sum = gs.map(g => va.weight * g.GT.gt).sum()")

    val resultsVSM = vdsGrouped.linreg(Array("sa.pheno"), "g.sum", covExpr = Array("sa.cov.Cov1", "sa.cov.Cov2"))
    val linregMap = resultsVSM.rowsTable().select("genes", "linreg.beta", "linreg.se", "linreg.tstat", "linreg.pval")
      .mapAnnotations { r => (r.getAs[String](0), (1 to 4).map{ i => Double.box(r.getAs[IndexedSeq[Double]](i)(0)) }) }
      .collect()
      .toMap
    val sampleMap = resultsVSM.rdd.map { case (keys, (_, gs)) =>
      val k = keys.asInstanceOf[Row].getAs[String](0)
      k -> gs.asInstanceOf[IndexedSeq[Row]].map { r => Double.box(r.getAs[Double](0)) }
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

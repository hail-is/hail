package is.hail.methods

import is.hail.SparkSuite
import org.testng.annotations.Test


class AnnotateAllelesSuite extends SparkSuite {
  @Test def test() {

    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .annotateAllelesExpr("va.gqMean = gs.map(g => g.gq).stats().mean," +
        "va.AC = gs.map(g => g.nNonRefAlleles()).sum()")
      .annotateVariantsExpr("va.callStatsAC = gs.callStats(g => v).AC[1:]")

    val testq = vds.queryVA("va.AC")._2
    val truthq = vds.queryVA("va.callStatsAC")._2

    vds.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(testq(va) == truthq(va))
      }

    val vds2 = vds.splitMulti()
      .annotateVariantsExpr("va.splitGqMean = gs.map(g => g.gq).stats().mean")

    val (_, testq2) = vds2.queryVA("va.gqMean")
    val (_, aIndexq) = vds2.queryVA("va.aIndex")
    val (_, truthq2) = vds2.queryVA("va.splitGqMean")

    vds2.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(
          testq2(va).asInstanceOf[IndexedSeq[Double]](
            aIndexq(va).asInstanceOf[Int] - 1) == truthq2(va))
      }
  }
}

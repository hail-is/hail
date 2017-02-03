package is.hail.methods

import is.hail.SparkSuite
import is.hail.driver._
import is.hail.io.vcf.LoadVCF
import org.testng.annotations.Test


class AnnotateAllelesSuite extends SparkSuite {
  @Test def test() {

    val vds = hc.importVCF("src/test/resources/sample2.vcf")
      .annotateAllelesExpr("va.gqMean = gs.map(g => g.gq).stats().mean," +
        "va.AC = gs.map(g => g.nNonRefAlleles).sum()", propagateGQ = true)
      .annotateVariantsExpr("va.callStatsAC = gs.callStats(g => v).AC[1:]")

    val testq = vds.queryVA("va.AC")._2
    val truthq = vds.queryVA("va.callStatsAC")._2

    vds.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(
          (testq(va), truthq(va)) match {
            case (Some(test), Some(truth)) =>
              test.asInstanceOf[IndexedSeq[Int]]
                .zip(truth.asInstanceOf[IndexedSeq[Int]])
                .forall({ case (x, y) => x == y })
            case (None, None) => true
            case _ => false
          }
        )
      }

    val vds2 = vds.splitMulti(propagateGQ = true)
      .annotateVariantsExpr("va.splitGqMean = gs.map(g => g.gq).stats().mean")

    val testq2 = vds2.queryVA("va.gqMean")._2
    val aIndexq = vds2.queryVA("va.aIndex")._2
    val truthq2 = vds2.queryVA("va.splitGqMean")._2

    vds2.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(
          (testq2(va), truthq2(va), aIndexq(va)) match {
            case (Some(test), Some(truth), Some(index)) =>
              test.asInstanceOf[IndexedSeq[Double]](index.asInstanceOf[Int] - 1) == truth.asInstanceOf[Double]
            case (None, None, _) => true
            case _ => false
          }
        )
      }
  }
}

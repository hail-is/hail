package is.hail.methods

import is.hail.SparkSuite
import is.hail.driver._
import is.hail.io.vcf.LoadVCF
import org.testng.annotations.Test


class AnnotateAllelesSuite extends SparkSuite {
  @Test def test(): Unit = {

    var s = State(sc, sqlContext, LoadVCF(sc, "src/test/resources/sample2.vcf"))
    s = AnnotateAlleles.run(s, Array("expr", "--propagate-gq", "-c",
      "va.gqMean = gs.map(g => g.gq).stats().mean," +
        "va.AC = gs.map(g => g.nNonRefAlleles).sum()"))

    s = AnnotateVariants.run(s, Array("expr", "-c",
      "va.callStatsAC = gs.callStats(g => v).AC[1:]"))

    val testq = s.vds.queryVA("va.AC")._2
    val truthq = s.vds.queryVA("va.callStatsAC")._2

    s.vds.variantsAndAnnotations
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

    s = SplitMulti.run(s, Array("--propagate-gq"))
    s = AnnotateVariants.run(s, Array("expr", "-c",
      "va.splitGqMean = gs.map(g => g.gq).stats().mean"))

    val testq2 = s.vds.queryVA("va.gqMean")._2
    val aIndexq = s.vds.queryVA("va.aIndex")._2
    val truthq2 = s.vds.queryVA("va.splitGqMean")._2

    s.vds.variantsAndAnnotations
      .collect()
      .foreach { case (v, va) =>
        assert(
          (testq2(va), truthq2(va), aIndexq(va)) match {
            case (Some(test), Some(truth), Some(index)) =>
              test.asInstanceOf[IndexedSeq[Double]](index.asInstanceOf[Int]-1) == truth.asInstanceOf[Double]
            case (None, None, _) => true
            case _ => false
          }
        )
      }
  }

}

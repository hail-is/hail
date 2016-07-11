package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.stats.LeveneHaldane
import org.testng.annotations.Test
import org.broadinstitute.hail.check._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant._

class HWESuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/HWE_test.vcf")

    val r = VariantQC.results(vds).map { case (v, a) => (v.start, a.HWEStats) }.collectAsMap()

    assert(r(1) ==(Some(0.0), 0.5))
    assert(r(2) ==(Some(0.25), 0.5))
    assert(D_==(r(3)._2, LeveneHaldane(4, 3).exactMidP(1)))
    assert(D_==(r(4)._2, LeveneHaldane(4, 4).exactMidP(2)))
    assert(D_==(r(5)._2, LeveneHaldane(3, 1).exactMidP(1)))
    assert(r(6) ==(None, 0.5))
  }

  @Test def testExpr() {
    val p = Prop.forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random)) { vds: VariantDataset =>
      val s = SplitMulti.run(State(sc, sqlContext, vds))
      val s2 = VariantQC.run(s)
      val s3 = AnnotateVariantsExpr.run(s2, Array("-c", "va.hweExpr = hwe(va.qc.nHomRef, va.qc.nHet, va.qc.nHomVar)"))
      val vds2 = s3.vds
      val (_, q1) = vds2.queryVA("va.hweExpr.rExpectedHetFrequency")
      val (_, q2) = vds2.queryVA("va.hweExpr.pHWE")
      val (_, q1e) = vds2.queryVA("va.qc.rExpectedHetFrequency")
      val (_, q2e) = vds2.queryVA("va.qc.pHWE")
      vds2
        .variantsAndAnnotations
        .map(_._2)
        .collect()
        .forall(a => q1(a) == q1e(a) && q2(a) == q2e(a))
    }
    p.check()
  }
}

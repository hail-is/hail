package org.broadinstitute.hail.vcf

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.{SplitMulti, State}
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.variant.{Genotype, VSMSubgen, Variant, VariantDataset, VariantSampleMatrix}
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils.simpleAssert

class SplitSuite extends SparkSuite {

  object Spec extends Properties("MultiSplit") {
    property("fakeRef implies wasSplit") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random)) { (vds: VariantDataset) =>
        var s = State(sc, sqlContext, vds)
        s = SplitMulti.run(s, Array[String]())
        val wasSplitQuerier = s.vds.vaSignature.query("wasSplit")
        s.vds.mapWithAll((v: Variant, va: Annotation, _: String, _: Annotation, g: Genotype) =>
          !g.fakeRef || wasSplitQuerier(va).asInstanceOf[Option[Boolean]].get)
          .collect()
          .forall(identity)
      }
  }

  @Test def minrep() {
    assert(SplitMulti.minRep(10, "TAA", "TA") == (10, "TA", "T"))
    assert(SplitMulti.minRep(10, "ACTG", "ACT") == (12, "TG", "T"))
    assert(SplitMulti.minRep(10, "AAACAAAC", "AAAC") == (10, "AAACA", "A"))
    assert(SplitMulti.minRep(10, "AATAA", "AAGAA") == (12, "T", "G"))
  }

  @Test def splitTest() {
    Spec.check()

    val vds1m = LoadVCF(sc, "src/test/resources/split_test.vcf")

    var s = State(sc, sqlContext)
    s = s.copy(vds = vds1m)
    s = SplitMulti.run(s, Array[String]())
    val vds1 = s.vds

    val vds2 = LoadVCF(sc, "src/test/resources/split_test_b.vcf")

    // test splitting and downcoding
    vds1.mapWithKeys((v, s, g) => ((v, s), g.copy(fakeRef = false)))
      .join(vds2.mapWithKeys((v, s, g) => ((v, s), g)))
      .foreach { case (k, (g1, g2)) =>
        if (g1 != g2)
          println(s"$g1, $g2")
        simpleAssert(g1 == g2) }

    val wasSplitQuerier = vds1.vaSignature.query("wasSplit")

    // test for wasSplit
    vds1.mapWithAll((v, va, s, sa, g) => (v.start, wasSplitQuerier(va).asInstanceOf[Option[Boolean]].get))
      .foreach { case (i, b) =>
        simpleAssert(b == (i != 1180))
      }

    // test for fakeRef
    assert(vds1.mapWithKeys((v, s, g) => ((v.start, v.alt, s), g.fakeRef)).filter(_._2).map(_._1.toString).collect.toSet
      == Set("(2167,AAAAC,HG00097)", "(2167,A,HG00100)", "(2167,AAAACAAAC,HG00103)", "(1183,C,HG00100)", "(2167,A,HG00103)", "(2167,AAAAC,HG00099)",
      "(2167,AAAAC,HG00104)", "(2167,AAAACAAAC,HG00104)", "(1783,TA,HG00100)", "(2167,A,HG00101)", "(2167,A,HG00099)", "(1183,C,HG00103)",
      "(2167,AAAAC,HG00103)", "(1183,C,HG00104)", "(1783,TA,HG00101)", "(2167,AAAACAAAC,HG00101)", "(2167,AAAAC,HG00101)", "(1783,T,HG00099)",
      "(2167,A,HG00097)", "(2167,AAAAC,HG00102)", "(2167,AAAACAAAC,HG00100)", "(1783,T,HG00101)", "(1783,TA,HG00102)", "(1783,T,HG00097)",
      "(2167,AAAACAAAC,HG00102)"))
  }
}

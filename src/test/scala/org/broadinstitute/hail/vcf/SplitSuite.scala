package org.broadinstitute.hail.vcf

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.{SplitMulti, State}
import org.broadinstitute.hail.methods.LoadVCF
import org.broadinstitute.hail.variant.{Genotype, VariantSampleMatrix, VariantDataset, Variant}
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.check.Prop._
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils.simpleAssert

class SplitSuite extends SparkSuite {

  object Spec extends Properties("MultiSplit") {
    property("fakeRef implies wasSplit") =
      forAll(VariantSampleMatrix.gen[Genotype](sc, Genotype.gen _)) { (vds: VariantDataset) =>
        var s = State(sc, sqlContext, vds)
        s = SplitMulti.run(s, Array[String]())
        val wasSplitQuerier = s.vds.vaSignature.query("wasSplit")
        s.vds.mapWithAll((v: Variant, va: Annotation, _: Int, g: Genotype) =>
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
      .foreach { case (k, (g1, g2)) => simpleAssert(g1 == g2) }

    val wasSplitQuerier = vds1.vaSignature.query("wasSplit")

    // test for wasSplit
    vds1.mapWithAll((v, va, s, g) => (v.start, wasSplitQuerier(va).asInstanceOf[Option[Boolean]].get))
      .foreach { case (i, b) =>
        simpleAssert(b == (i != 1180))
      }

    // test for fakeRef
    assert(vds1.mapWithKeys((v, s, g) => ((v.start, v.alt, s), g.fakeRef)).filter(_._2).map(_._1.toString).collect.toSet
      == Set("(2167,AAAAC,1)", "(2167,A,3)", "(2167,AAAACAAAC,6)", "(1183,C,3)", "(2167,A,6)", "(2167,AAAAC,2)",
      "(2167,AAAAC,7)", "(2167,AAAACAAAC,7)", "(1783,TA,3)", "(2167,A,4)", "(2167,A,2)", "(1183,C,6)",
      "(2167,AAAAC,6)", "(1183,C,7)", "(1783,TA,4)", "(2167,AAAACAAAC,4)", "(2167,AAAAC,4)", "(1783,T,2)",
      "(2167,A,1)", "(2167,AAAAC,5)", "(2167,AAAACAAAC,3)", "(1783,T,4)", "(1783,TA,5)", "(1783,T,1)",
      "(2167,AAAACAAAC,5)"))

  }
}

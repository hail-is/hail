package org.broadinstitute.hail.vcf

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{MultiSplit, State}
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
        s = MultiSplit.run(s, Array[String]())
        s.vds.mapWithKeys((v: Variant, _: Int, g: Genotype) =>
          !g.fakeRef || v.wasSplit)
          .collect()
          .forall(identity)
      }
  }

  @Test def splitTest() {
    Spec.check()

    val vds1m = LoadVCF(sc, "src/test/resources/split_test.vcf")

    var s = State(sc, sqlContext)
    s = s.copy(vds = vds1m)
    s = MultiSplit.run(s, Array[String]())
    val vds1 = s.vds

    val vds2 = LoadVCF(sc, "src/test/resources/split_test_b.vcf")

    // test splitting and downcoding
    vds1.mapWithKeys((v, s, g) => ((v.copy(wasSplit = false), s), g.copy(fakeRef = false)))
      .join(vds2.mapWithKeys((v, s, g) => ((v, s), g)))
      .foreach { case (k, (g1, g2)) => simpleAssert(g1 == g2) }

    // test for wasSplit
    vds1.mapWithKeys((v, s, g) => (v.start, v.wasSplit)).foreach { case (i, b) => simpleAssert(b == (i != 1180)) }

    // test for fakeRef
    assert(vds1.mapWithKeys((v, s, g) => ((v.start, v.alt, s), g.fakeRef)).filter(_._2).map(_._1.toString).collect.toSet
      == Set("(2167,AAAACAAAC,1)", "(2167,A,3)", "(2167,AAAACAAACAAAC,6)", "(1183,C,3)", "(2167,A,6)", "(2167,AAAACAAAC,2)",
      "(2167,AAAACAAAC,7)", "(2167,AAAACAAACAAAC,7)", "(1783,TAA,3)", "(2167,A,4)", "(2167,A,2)", "(1183,C,6)",
      "(2167,AAAACAAAC,6)", "(1183,C,7)", "(1783,TAA,4)", "(2167,AAAACAAACAAAC,4)", "(2167,AAAACAAAC,4)", "(1783,T,2)",
      "(2167,A,1)", "(2167,AAAACAAAC,5)", "(2167,AAAACAAACAAAC,3)", "(1783,T,4)", "(1783,TAA,5)", "(1783,T,1)",
      "(2167,AAAACAAACAAAC,5)"))

  }
}

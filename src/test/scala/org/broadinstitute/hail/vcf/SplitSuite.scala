package org.broadinstitute.hail.vcf

import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.methods.LoadVCF
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils.simpleAssert

class SplitSuite extends SparkSuite {
  @Test def SplitTest() {

    val file1 = "src/test/resources/split_test.vcf"
    val file2 = "src/test/resources/split_test_b.vcf"

    val vds1 = LoadVCF(sc, file1)
    val vds2 = LoadVCF(sc, file2)

    // test splitting and downcoding
    vds1.mapWithKeys((v, s, g) => ((v.copy(wasSplit = false), s), g.copy(fakeRef = false)))
      .join(vds2.mapWithKeys((v, s, g) => ((v, s), g)))
      .foreach { case (k, (g1,  g2)) =>
        if (g1.isNotCalled && g2.isNotCalled)
          None
        else
          simpleAssert(g1 == g2)
      }

    // test for wasSplit
    vds1.mapWithKeys((v, s, g) => (v.start, v.wasSplit)).foreach{case (i, b) => simpleAssert(b == (i != 1180))}

    // test for fakeRef
    assert(vds1.mapWithKeys((v, s, g) => ((v.start, v.alt, s), g.fakeRef)).filter(_._2).map(_._1.toString).collect.toSet
      == Set("(2167,AAAACAAAC,1)", "(2167,A,3)", "(2167,AAAACAAACAAAC,6)", "(1183,C,3)", "(2167,A,6)", "(2167,AAAACAAAC,2)",
      "(2167,AAAACAAAC,7)", "(2167,AAAACAAACAAAC,7)", "(1783,TAA,3)", "(2167,A,4)", "(2167,A,2)", "(1183,C,6)",
      "(2167,AAAACAAAC,6)", "(1183,C,7)", "(1783,TAA,4)", "(2167,AAAACAAACAAAC,4)", "(2167,AAAACAAAC,4)", "(1783,T,2)",
      "(2167,A,1)", "(2167,AAAACAAAC,5)", "(2167,AAAACAAACAAAC,3)", "(1783,T,4)", "(1783,TAA,5)", "(1783,T,1)",
      "(2167,AAAACAAACAAAC,5)"))

  }
}

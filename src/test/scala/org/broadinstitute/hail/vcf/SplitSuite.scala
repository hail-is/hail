package org.broadinstitute.hail.vcf

import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.methods.LoadVCF
import org.testng.annotations.Test


class SplitSuite extends SparkSuite {
  @Test def SplitTest() {

    val file1 = "src/test/resources/split_test.vcf"
    val file2 = "src/test/resources/split_test_b.vcf"

    val vds1 = LoadVCF(sc, file1)
    val vds2 = LoadVCF(sc, file2)

//    vds1.mapWithKeys((v, s, g) => ((v, s), g))
//      .join(vds2.mapWithKeys((v, s, g) => ((v, s), g)))
//      .foreach { case (k, (g1,  g2)) =>
//        if (g1.isNotCalled && g2.isNotCalled)
//          None
//        else if (g1.isCalled && g2.isCalled
//          && g1 == g2)
//          None
//        else
//          println(s"differ: at $k: $g1 vs $g2")
//      }
//

  }
}

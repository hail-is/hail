package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{SplitMulti, State, VariantQC}
import org.testng.annotations.Test

class MapReduceSuite extends SparkSuite {
  @Test def testVariants(): Unit = {

    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = VariantQC.run(s, Array.empty[String])


  }
}

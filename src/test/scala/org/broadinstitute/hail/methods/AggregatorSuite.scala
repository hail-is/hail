package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.{FilterVariants, SplitMulti, State}
import org.testng.annotations.Test

class AggregatorSuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/sample2.vcf")
    var s = SplitMulti.run(State(sc, sqlContext, vds), Array.empty[String])
    s = FilterVariants.run(s, Array("--keep", "-c", "gs.count(g.gq >= 20) >= 2000"))
    println(s.vds.nVariants)
  }

}

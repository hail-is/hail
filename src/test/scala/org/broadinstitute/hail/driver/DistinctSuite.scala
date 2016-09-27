package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class DistinctSuite extends SparkSuite {

  @Test def test() {
    val s = ImportVCF.run(State(sc, sqlContext, null),
      Array("src/test/resources/sample.vcf", "src/test/resources/sample.vcf"))
    Distinct.run(s).vds.rdd.count()

    assert(Distinct.DistinctReport.accumulator.value == 346L)
  }
}

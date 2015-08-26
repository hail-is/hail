package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.driver.VariantQC
import org.broadinstitute.hail.stats.LeveneHaldane
import org.testng.annotations.Test
import org.broadinstitute.hail.Utils._

class HWESuite extends SparkSuite {

  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/HWE_test.vcf")

    val r = VariantQC.results(vds).map { case (v, a) => (v.start, a.levineHaldaneStats) }.collectAsMap()

    assert(r(1) == (Some(0.0), 0.5))
    assert(r(2) == (Some(0.25), 0.5))
    assert(D_==(r(3)._2, LeveneHaldane(4, 3).exactMidP(1)))
    assert(D_==(r(4)._2, LeveneHaldane(4, 4).exactMidP(2)))
    assert(D_==(r(5)._2, LeveneHaldane(3, 1).exactMidP(1)))
    assert(r(6) == (None, 0.5))
  }
}

package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class GQByDPBinSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "src/test/resources/gqbydp_test.vcf")
    val gqbydp = GQByDPBins(vds)
    assert(gqbydp == Map((0, 5) -> 0.5, (1, 2) -> 0.0))
  }
}

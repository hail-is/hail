package org.broadinstitute.k3.methods

import org.apache.spark.{SparkContext, SparkConf}
import org.broadinstitute.k3.SparkSuite
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GQByDPBinSuite extends SparkSuite {
  @Test def test() {
    val vds = LoadVCF(sc, "sparky", "src/test/resources/gqbydp_test.vcf")
    val gqbydp = GQByDPBins(vds)
    assert(gqbydp == Map((0, 5) -> 0.5, (1, 2) -> 0.0))
  }
}

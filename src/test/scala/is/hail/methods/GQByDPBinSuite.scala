package is.hail.methods

import is.hail.SparkSuite
import is.hail.io.vcf.LoadVCF
import org.testng.annotations.Test

class GQByDPBinSuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/gqbydp_test.vcf")
    val gqbydp = GQByDPBins(vds)
    assert(gqbydp == Map(("1", 5) -> 0.5, ("2", 2) -> 0.0))
  }
}

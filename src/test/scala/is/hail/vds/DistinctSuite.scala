package is.hail.vds

import is.hail.SparkSuite
import org.testng.annotations.Test

class DistinctSuite extends SparkSuite {

  @Test def test() {
    assert(hc.importVCFs(List("src/test/resources/sample.vcf", "src/test/resources/sample.vcf"))
      .deduplicate().countVariants() == 346)
  }
}

package is.hail.vds

import is.hail.SparkSuite
import is.hail.methods.DuplicateReport
import org.testng.annotations.Test

class DistinctSuite extends SparkSuite {

  @Test def test() {
    hc.importVCFs(List("src/test/resources/sample.vcf", "src/test/resources/sample.vcf"))
      .deduplicate().count()

    assert(DuplicateReport.accumulator.value._1 == 346L)
  }
}

package is.hail.vds

import is.hail.SparkSuite
import org.testng.annotations.Test

class GenotypeTableSuite extends SparkSuite {
  @Test def test() {
    val vds = hc.importVCF("src/test/resources/sample.vcf")

    vds.entriesTable().typeCheck()

    assert(vds.entriesTable().rdd.count() == vds.countVariants() * vds.numCols)
  }
}

package is.hail.vds

import is.hail.SparkSuite
import org.testng.annotations.Test

class JoinSuite extends SparkSuite {
  @Test def test() {
    val joined = hc.importVCF("src/test/resources/joined.vcf")

    val joinedPath = tmpDir.createTempFile("joined", "vds")


    val left = hc.importVCF("src/test/resources/joinleft.vcf")
    val right = hc.importVCF("src/test/resources/joinright.vcf")

    // make sure joined VDS writes
    left.join(right).write(joinedPath)

    assert(joined.same(hc.readVDS(joinedPath)))
  }
}

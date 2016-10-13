package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class JoinSuite extends SparkSuite {
  @Test def test() {
    var sjoined = State(sc, sqlContext)
    sjoined = ImportVCF.run(sjoined, Array("src/test/resources/joined.vcf"))

    var s = State(sc, sqlContext)

    val joinedPath = tmpDir.createTempFile("joined", "vds")

    s = ImportVCF.run(s, Array("src/test/resources/joinright.vcf"))
    s = Put.run(s, Array("-n", "right"))

    s = ImportVCF.run(s, Array("src/test/resources/joinleft.vcf"))

    s = Join.run(s, Array("-r", "right"))

    // make sure joined VDS writes
    s = Write.run(s, Array("-o", joinedPath))

    assert(sjoined.vds.same(s.vds))
  }
}

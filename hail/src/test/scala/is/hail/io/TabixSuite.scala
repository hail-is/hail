package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.io.tabix.TabixReader

class TabixSuite extends SparkSuite {
  val vcfFile = "src/test/resources/trioDup.vcf"
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  lazy val reader = TabixReader(vcfGzFile)

  @Test def testSequenceNames() {
    // TODO Implement
  }

  @Test def testSequenceSet() {
    // TODO Implement
  }
}

package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.io.tabix.TabixReader
import is.hail.testUtils._

import org.testng.annotations.{BeforeTest, Test}

class TabixSuite extends SparkSuite {
  val vcfFile = "src/test/resources/trioDup.vcf"
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  lazy val reader = TabixReader(vcfGzFile)

  @BeforeTest def initialize() {
    hc // reference to initialize
  }

  @Test def testSequenceNames() {
    val expectedSeqNames = new Array[String](24);
    for (i <- 1 to 22) {
      expectedSeqNames(i-1) = i.toString
    }
    expectedSeqNames(22) = "X";
    expectedSeqNames(23) = "Y";

    val sequenceNames = reader.index.chr2tid.keySet
    assert(expectedSeqNames.length == sequenceNames.size)
    for (s <- expectedSeqNames) {
      assert(sequenceNames.contains(s), s"sequencNames does not contain ${ s }")
    }
  }

  @Test def testSequenceSet() {
    // TODO Implement
  }
}

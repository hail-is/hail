package is.hail.io

import is.hail.{SparkSuite, TestUtils}
import is.hail.io.tabix._
import is.hail.testUtils._

import htsjdk.tribble.readers.{TabixReader => HtsjdkTabixReader}

import org.testng.asserts.SoftAssert
import org.testng.annotations.{BeforeTest, Test}

class TabixSuite extends SparkSuite {
  val vcfFile = "src/test/resources/trioDup.vcf"
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  lazy val reader = new TabixReader(vcfGzFile)

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

    val asserts = new SoftAssert()
    for (s <- expectedSeqNames) {
      asserts.assertTrue(sequenceNames.contains(s), s"sequenceNames does not contain ${ s }")
    }
    asserts.assertAll()
  }

  @Test def testSequenceSet() {
    val chrs = reader.index.chr2tid.keySet
    assert(!chrs.isEmpty)
    assert(chrs.contains("1"))
    assert(!chrs.contains("MT"))
  }

  @Test def testLineIterator() {
    val htsjdkrdr = new HtsjdkTabixReader(vcfGzFile)
    // In range access
    for (chr <- Seq("1", "19", "X")) {
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 1, 400);
      val hailIter = new TabixLineIterator(reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assert(hailStr == htsStr)
      assert(hailIter.next() == null)
    }

    // Out of range access
    for (chr <- Seq("1", "19", "X")) {
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 400, 400);
      val hailIter = new TabixLineIterator(reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 400, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assert(hailStr == null)
      assert(hailStr == htsStr)
    }
  }

  @Test def testLineIterator2() {
    val vcfFile = "src/test/resources/sample.vcf.gz"
    val chr = "20"
    val htsjdkrdr = new HtsjdkTabixReader(vcfFile)
    val hailrdr = new TabixReader(vcfFile)
    val tid = hailrdr.chr2tid(chr)

    // One approximate interval, one (almost) exact interval, there is a slight difference in
    // behavior between htsjdk and hail in that htsjdk will query the half closed interval
    // (beg,end], whereas hail will query the closed interval [beg,end]
    for ((start, end) <- Seq(10570000 -> 13000000, 10019092 -> 16360860)) {
      val pairs = hailrdr.queryPairs(tid, start, end)
      val htsIter = htsjdkrdr.query(chr, start, end)
      val hailIter = new TabixLineIterator(hailrdr.filePath, pairs)
      var hailStr = hailIter.next()
      while (hailStr != null) {
        val htsStr = htsIter.next()
        assert(htsStr != null)
        assert(htsStr == hailStr)

        hailStr = hailIter.next()
      }
      assert(htsIter.next() == null)
    }
  }
}

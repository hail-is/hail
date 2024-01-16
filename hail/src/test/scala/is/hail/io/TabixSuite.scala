package is.hail.io

import is.hail.HailSuite
import is.hail.io.tabix._
import is.hail.io.vcf.TabixVCF

import htsjdk.tribble.readers.{TabixReader => HtsjdkTabixReader}
import org.testng.annotations.{BeforeTest, Test}
import org.testng.asserts.SoftAssert

class TabixSuite extends HailSuite {
  // use .gz for several tests and .bgz for another to test handling of both
  // extensions.
  val vcfFile = "src/test/resources/trioDup.vcf"
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  lazy val reader = new TabixReader(vcfGzFile, fs)

  @BeforeTest def initialize(): Unit = {
    hc // reference to initialize
  }

  @Test def testLargeNumberOfSequences(): Unit = {
    val tbx = new TabixReader(null, fs, Some("src/test/resources/large-tabix.tbi"))
    // known length of sequences
    assert(tbx.index.seqs.length == 3366)
  }

  @Test def testSequenceNames(): Unit = {
    val expectedSeqNames = new Array[String](24);
    for (i <- 1 to 22)
      expectedSeqNames(i - 1) = i.toString
    expectedSeqNames(22) = "X";
    expectedSeqNames(23) = "Y";

    val sequenceNames = reader.index.chr2tid.keySet
    assert(expectedSeqNames.length == sequenceNames.size)

    val asserts = new SoftAssert()
    for (s <- expectedSeqNames)
      asserts.assertTrue(sequenceNames.contains(s), s"sequenceNames does not contain $s")
    asserts.assertAll()
  }

  @Test def testSequenceSet(): Unit = {
    val chrs = reader.index.chr2tid.keySet
    assert(!chrs.isEmpty)
    assert(chrs.contains("1"))
    assert(!chrs.contains("MT"))
  }

  @Test def testLineIterator(): Unit = {
    val htsjdkrdr = new HtsjdkTabixReader(vcfGzFile)
    // In range access
    for (chr <- Seq("1", "19", "X")) {
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 1, 400);
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assert(hailStr == htsStr)
      assert(hailIter.next() == null)
    }

    // Out of range access
    // NOTE: We use the larger interval for the htsjdk iterator because the
    // hail iterator may return the one record that is contained in each of the
    // chromosomes we check
    for (chr <- Seq("1", "19", "X")) {
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 350, 400);
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      if (hailStr != null)
        assert(hailStr == htsStr)
      assert(hailIter.next() == null)
    }

    // beg == end
    for (chr <- Seq("1", "19", "X")) {
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 100, 100);
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 100, 100);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assert(hailStr == null)
      assert(hailStr == htsStr)
    }
  }

  def _testLineIterator2(vcfFile: String): Unit = {
    val chr = "20"
    val htsjdkrdr = new HtsjdkTabixReader(vcfFile)
    val hailrdr = new TabixReader(vcfFile, fs)
    val tid = hailrdr.chr2tid(chr)

    for (
      (start, end) <-
        Seq(
          (12990058, 12990059), // Small interval, containing just one locus at end
          (10570000, 13000000), // Open interval
          (10019093, 16360860), // Closed interval
          (11000000, 13029764), // Half open (beg, end]
          (17434340, 18000000), // Half open [beg, end)
          (13943975, 14733634), // Some random intervals
          (11578765, 15291865),
          (12703588, 16751726),
        )
    ) {
      val pairs = hailrdr.queryPairs(tid, start, end)
      val htsIter = htsjdkrdr.query(chr, start, end)
      val hailIter = new TabixLineIterator(fs, hailrdr.filePath, pairs)
      var htsStr = htsIter.next()
      var test = false
      while (htsStr != null) {
        val hailStr = hailIter.next()
        if (test) {
          assert(hailStr == htsStr, s"\nhail   : $hailStr\nhtsjdk : $htsStr")
          htsStr = htsIter.next()
        } else if (hailStr == htsStr) {
          htsStr = htsIter.next()
          test = true
        } else {
          assert(hailStr != null)
        }
      }
    }
  }

  @Test def testLineIterator2(): Unit = {
    _testLineIterator2("src/test/resources/sample.vcf.bgz")
  }

  @Test def testWriter(): Unit = {
    val vcfFile = "src/test/resources/sample.vcf.bgz"
    val path = ctx.createTmpPath("test-tabix-write", "bgz")
    fs.copy(vcfFile, path)
    TabixVCF(fs, vcfFile)
    _testLineIterator2(vcfFile)
  }
}

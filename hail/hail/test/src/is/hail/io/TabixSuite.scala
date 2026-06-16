package is.hail.io

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.io.fs.FS
import is.hail.io.tabix._
import is.hail.io.vcf.TabixVCF

import htsjdk.tribble.readers.{TabixReader => HtsjdkTabixReader}
import org.junit.jupiter.api.{BeforeAll, Test}
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.convertToAnyShouldWrapper

class TabixSuite {
  // use .gz for several tests and .bgz for another to test handling of both
  // extensions.
  val vcfFile = getTestResource("trioDup.vcf")
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  private var reader: TabixReader = _

  @BeforeAll def initReader(fs: FS): Unit =
    reader = new TabixReader(vcfGzFile, fs)

  @Test def testLargeNumberOfSequences(implicit ctx: ExecuteContext): Unit = {
    val tbx = new TabixReader(null, ctx.fs, Some(getTestResource("large-tabix.tbi")))
    // known length of sequences
    assertEq(tbx.index.seqs.length, 3366)
  }

  @Test def testSequenceNames(): Unit = {
    val expectedSeqNames = new Array[String](24);
    for (i <- 1 to 22)
      expectedSeqNames(i - 1) = i.toString
    expectedSeqNames(22) = "X";
    expectedSeqNames(23) = "Y";

    val sequenceNames = reader.index.chr2tid.keySet
    assertEq(expectedSeqNames.length, sequenceNames.size)
    sequenceNames should contain theSameElementsAs expectedSeqNames
  }

  @Test def testSequenceSet(): Unit = {
    val chrs = reader.index.chr2tid.keySet
    assert(chrs.nonEmpty)
    assert(chrs.contains("1"))
    assert(!chrs.contains("MT"))
  }

  @Test def testLineIterator(implicit ctx: ExecuteContext): Unit = {
    val htsjdkrdr = new HtsjdkTabixReader(vcfGzFile)
    // In range access
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 1, 400);
      val hailIter = new TabixLineIterator(ctx.fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assertEq(hailStr, htsStr)
      assertEq(hailIter.next(), null)
    }

    // Out of range access
    // NOTE: We use the larger interval for the htsjdk iterator because the
    // hail iterator may return the one record that is contained in each of the
    // chromosomes we check
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 350, 400);
      val hailIter = new TabixLineIterator(ctx.fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      if (hailStr != null)
        assertEq(hailStr, htsStr)
      assertEq(hailIter.next(), null)
    }

    // beg == end
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 100, 100);
      val hailIter = new TabixLineIterator(ctx.fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 100, 100);
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assertEq(hailStr, null)
      assertEq(hailStr, htsStr)
    }
  }

  def _testLineIterator2(vcfFile: String)(implicit ctx: ExecuteContext): Unit = {
    val chr = "20"
    val htsjdkrdr = new HtsjdkTabixReader(vcfFile)
    val hailrdr = new TabixReader(vcfFile, ctx.fs)
    val tid = hailrdr.chr2tid(chr)

    Seq(
      (12990058, 12990059), // Small interval, containing just one locus at end
      (10570000, 13000000), // Open interval
      (10019093, 16360860), // Closed interval
      (11000000, 13029764), // Half open (beg, end]
      (17434340, 18000000), // Half open [beg, end)
      (13943975, 14733634), // Some random intervals
      (11578765, 15291865),
      (12703588, 16751726),
    ).foreach { case (start, end) =>
      val pairs = hailrdr.queryPairs(tid, start, end)
      val htsIter = htsjdkrdr.query(chr, start, end)
      val hailIter = new TabixLineIterator(ctx.fs, hailrdr.filePath, pairs)
      var htsStr = htsIter.next()
      var test = false
      while (htsStr != null) {
        val hailStr = hailIter.next()
        if (test) {
          assertEq(hailStr, htsStr, s"\nhail   : $hailStr\nhtsjdk : $htsStr")
          htsStr = htsIter.next()
        } else if (hailStr == htsStr) {
          htsStr = htsIter.next()
          test = true
        } else {
          assert(hailStr != null)
          ()
        }
      }
    }
  }

  @Test def testLineIterator2(implicit ctx: ExecuteContext): Unit =
    _testLineIterator2(getTestResource("sample.vcf.bgz"))

  @Test def testWriter(implicit ctx: ExecuteContext): Unit = {
    val vcfFile = getTestResource("sample.vcf.bgz")
    val path = ctx.createTmpPath("test-tabix-write", "bgz")
    ctx.fs.copy(vcfFile, path)
    TabixVCF(ctx.fs, vcfFile)
    _testLineIterator2(vcfFile)
  }
}

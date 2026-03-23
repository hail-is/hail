package is.hail.io

import is.hail.HailSuite
import is.hail.io.tabix._
import is.hail.io.vcf.TabixVCF

import htsjdk.tribble.readers.{TabixReader => HtsjdkTabixReader}

class TabixSuite extends HailSuite {
  // use .gz for several tests and .bgz for another to test handling of both
  // extensions.
  val vcfFile = getTestResource("trioDup.vcf")
  val vcfGzFile = vcfFile + ".gz"
  val vcfGzTbiFile = vcfGzFile + ".tbi"

  lazy val reader = new TabixReader(vcfGzFile, fs)

  test("LargeNumberOfSequences") {
    val tbx = new TabixReader(null, fs, Some(getTestResource("large-tabix.tbi")))
    // known length of sequences
    assertEquals(tbx.index.seqs.length, 3366)
  }

  test("SequenceNames") {
    val expectedSeqNames = new Array[String](24)
    for (i <- 1 to 22)
      expectedSeqNames(i - 1) = i.toString
    expectedSeqNames(22) = "X"
    expectedSeqNames(23) = "Y"

    val sequenceNames = reader.index.chr2tid.keySet
    assertEquals(sequenceNames.size, expectedSeqNames.length)
    assertEquals(sequenceNames, expectedSeqNames.toSet)
  }

  test("SequenceSet") {
    val chrs = reader.index.chr2tid.keySet
    assert(chrs.nonEmpty)
    assert(chrs.contains("1"))
    assert(!chrs.contains("MT"))
  }

  test("LineIterator") {
    val htsjdkrdr = new HtsjdkTabixReader(vcfGzFile)
    // In range access
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 1, 400)
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400)
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assertEquals(hailStr, htsStr)
      assert(hailIter.next() == null)
    }

    // Out of range access
    // NOTE: We use the larger interval for the htsjdk iterator because the
    // hail iterator may return the one record that is contained in each of the
    // chromosomes we check
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 350, 400)
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 1, 400)
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      if (hailStr != null)
        assertEquals(hailStr, htsStr)
      assert(hailIter.next() == null)
    }

    // beg == end
    Seq("1", "19", "X").foreach { chr =>
      val tid = reader.chr2tid(chr)
      val pairs = reader.queryPairs(tid, 100, 100)
      val hailIter = new TabixLineIterator(fs, reader.filePath, pairs)
      val htsIter = htsjdkrdr.query(chr, 100, 100)
      val hailStr = hailIter.next()
      val htsStr = htsIter.next()
      assert(hailStr == null)
      assertEquals(hailStr, htsStr)
    }
  }

  def _testLineIterator2(vcfFile: String): Unit = {
    val chr = "20"
    val htsjdkrdr = new HtsjdkTabixReader(vcfFile)
    val hailrdr = new TabixReader(vcfFile, fs)
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
          ()
        }
      }
    }
  }

  test("LineIterator2")(_testLineIterator2(getTestResource("sample.vcf.bgz")))

  test("Writer") {
    val vcfFile = getTestResource("sample.vcf.bgz")
    val path = ctx.createTmpPath("test-tabix-write", "bgz")
    fs.copy(vcfFile, path)
    TabixVCF(fs, vcfFile)
    _testLineIterator2(vcfFile)
  }
}

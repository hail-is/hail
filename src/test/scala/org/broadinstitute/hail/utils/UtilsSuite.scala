package org.broadinstitute.hail.utils

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.SpanningIterator
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import org.broadinstitute.hail.Utils._

class UtilsSuite extends SparkSuite {
  @Test def testD_==() {
    assert(D_==(1, 1))
    assert(D_==(1, 1 + 1E-7))
    assert(!D_==(1, 1 + 1E-5))
    assert(D_==(1E10, 1E10 + 1))
    assert(!D_==(1E-10, 2E-10))
    assert(D_==(0.0, 0.0))
    assert(D_!=(1E-307, 0.0))
    assert(D_==(1E-308, 0.0))
    assert(D_==(1E-320, -1E-320))
  }

  @Test def testFlushDouble() {
    assert(flushDouble(8.0E-308) == 8.0E-308)
    assert(flushDouble(-8.0E-308) == -8.0E-308)
    assert(flushDouble(8.0E-309) == 0.0)
    assert(flushDouble(-8.0E-309) == 0.0)
    assert(flushDouble(0.0) == 0.0)
  }

  @Test def testAreDistinct() {
    assert(Array().areDistinct())
    assert(Array(1).areDistinct())
    assert(Array(1,2).areDistinct())
    assert(!Array(1,1).areDistinct())
    assert(!Array(1,2,1).areDistinct())
  }

  @Test def testIsIncreasing() {
    assert(Seq[Int]().isIncreasing)
    assert(Seq(1).isIncreasing)
    assert(Seq(1,2).isIncreasing)
    assert(! Seq(1,1).isIncreasing)
    assert(! Seq(1,2,1).isIncreasing)

    assert(Array(1,2).isIncreasing)
  }

  @Test def testIsSorted() {
    assert(Seq[Int]().isSorted)
    assert(Seq(1).isSorted)
    assert(Seq(1,2).isSorted)
    assert(Seq(1,1).isSorted)
    assert(! Seq(1,2,1).isSorted)

    assert(Array(1,1).isSorted)
  }

  @Test def testHadoopStripCodec() {
    assert(hadoopStripCodec("file.tsv", sc.hadoopConfiguration) == "file.tsv")
    assert(hadoopStripCodec("file.tsv.gz", sc.hadoopConfiguration) == "file.tsv")
    assert(hadoopStripCodec("file.tsv.bgz", sc.hadoopConfiguration) == "file.tsv")
    assert(hadoopStripCodec("file.tsv.lz4", sc.hadoopConfiguration) == "file.tsv")
    assert(hadoopStripCodec("file", sc.hadoopConfiguration) == "file")
  }

  @Test def spanningIterator() = {
    assert(span(List()) == List())
    assert(span(List((1, "a"))) == List((1, List("a"))))
    assert(span(List((1, "a"), (1, "b"))) == List((1, List("a", "b"))))
    assert(span(List((1, "a"), (2, "b"))) == List((1, List("a")), (2, List("b"))))
    assert(span(List((1, "a"), (1, "b"), (2, "c"))) ==
      List((1, List("a", "b")), (2, List("c"))))
    assert(span(List((1, "a"), (2, "b"), (2, "c"))) ==
      List((1, List("a")), (2, List("b", "c"))))
    assert(span(List((1, "a"), (2, "b"), (1, "c"))) ==
      List((1, List("a")), (2, List("b")), (1, List("c"))))
  }

  def span[K, V](tuples: List[(K, V)]) = {
    new SpanningIterator(tuples.iterator).toIterable.toList
  }
}

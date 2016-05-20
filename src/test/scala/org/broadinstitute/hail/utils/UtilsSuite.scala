package org.broadinstitute.hail.utils

import org.broadinstitute.hail.SparkSuite
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

  @Test def testPairRDDNoDup() {
    val pairRDD1 = sc.parallelize(Array(1, 2, 3, 4)).map{i => (i, i*i)}
    val pairRDD2 = sc.parallelize(Array(1, 2, 3, 4)).map{i => (i, i)}

    val join = pairRDD1.leftOuterJoin(pairRDD1)
    assert(join.count() == 4)
  }
}

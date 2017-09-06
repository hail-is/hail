package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class RichRDDSuite extends SparkSuite {
  @Test def testTakeByPartition() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)
    assert(r.headPerPartition(5).count() == 100)
  }

  @Test def testHead() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)
    val partitionRanges = r.countPerPartition().scanLeft(Range(0, 0)) { case (x, c) => Range(x.end, x.end + c.toInt) }

    def getExpectedNumPartitions(n: Int): Int =
      partitionRanges.indexWhere(_.contains(math.max(0, n - 1)))

    for (n <- Array(0, 15, 200, 562, 1024, 2000)) {
      val t = r.head(n)
      val nActual = math.min(n, 1024)

      assert(t.collect() sameElements (0 until nActual))
      assert(t.count() == nActual)
      assert(t.getNumPartitions == getExpectedNumPartitions(nActual))
    }
  }
}

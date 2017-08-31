package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class RichRDDSuite extends SparkSuite {
  @Test def testTakeByPartition() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)
    assert(r.headPerPartition(5).count() == 100)
  }

  @Test def testTake() {
    val r = sc.parallelize(0 until 1024, numSlices = 20)

    val t1 = r.head(15)
    assert(t1.count == 15)

    val t2 = r.head(200)
    assert(t2.count == 200)

    val t3 = r.head(562)
    assert(t3.count == 562)

    val t4 = r.head(2000)
    assert(t4.count == 1024)

    val t5 = r.head(0)
    assert(t5.count == 0)
  }
}

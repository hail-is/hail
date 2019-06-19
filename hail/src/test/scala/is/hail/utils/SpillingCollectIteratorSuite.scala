package is.hail.utils

import is.hail.SparkSuite
import org.testng.annotations.Test

class SpillingCollectIteratorSuite extends SparkSuite {
  @Test def addOneElement() {
    val array = (0 to 100000).toArray
    val sci = SpillingCollectIterator(sc.parallelize(array, 10000), 100)
    assert(sci.hasNext)
    assert(sci.next() == 0)
    assert(sci.hasNext)
    assert(sci.next() == 1)
    assert(sci.toArray == (2 to 100000))
  }
}

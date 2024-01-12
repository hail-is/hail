package is.hail.utils

import is.hail.HailSuite

import org.testng.annotations.Test

class SpillingCollectIteratorSuite extends HailSuite {
  @Test def addOneElement() {
    val array = (0 to 1234).toArray
    val sci = SpillingCollectIterator(ctx.localTmpdir, fs, sc.parallelize(array, 99), 100)
    assert(sci.hasNext)
    assert(sci.next() == 0)
    assert(sci.hasNext)
    assert(sci.next() == 1)
    assert(sci.toArray sameElements (2 to 1234))
  }
}

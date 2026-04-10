package is.hail.utils

import is.hail.HailSuite

class SpillingCollectIteratorSuite extends HailSuite {
  test("addOneElement") {
    val array = (0 to 1234)
    val sci = SpillingCollectIterator(ctx.localTmpdir, fs, sc.parallelize(array, 99), 100)
    assert(sci.hasNext)
    assertEquals(sci.next(), 0)
    assert(sci.hasNext)
    assertEquals(sci.next(), 1)
    assert(sci.toArray sameElements (2 to 1234))
  }
}

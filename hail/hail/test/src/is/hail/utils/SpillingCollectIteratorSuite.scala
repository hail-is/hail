package is.hail.utils

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext

import org.junit.jupiter.api.Test

class SpillingCollectIteratorSuite {
  @Test def addOneElement(implicit ctx: ExecuteContext): Unit = {
    val array = (0 to 1234)
    val sci = SpillingCollectIterator(
      ctx.localTmpdir,
      ctx.fs,
      ctx.backend.asSpark.sc.parallelize(array, 99),
      100,
    )
    assert(sci.hasNext)
    assertEq(sci.next(), 0)
    assert(sci.hasNext)
    assertEq(sci.next(), 1)
    assert(sci.toArray sameElements (2 to 1234))
  }
}

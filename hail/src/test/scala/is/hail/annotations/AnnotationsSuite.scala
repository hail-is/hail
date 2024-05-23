package is.hail.annotations

import is.hail.HailSuite

import org.testng.annotations.Test

/** This testing suite evaluates the functionality of the [[is.hail.annotations]] package */
class AnnotationsSuite extends HailSuite {
  @Test def testExtendedOrdering(): Unit = {
    val ord = ExtendedOrdering.extendToNull(implicitly[Ordering[Int]])
    val rord = ord.reverse

    assert(ord.lt(5, 7))
    assert(ord.lt(5, null))
    assert(ord.gt(null, 7))
    assert(ord.equiv(3, 3))
    assert(ord.equiv(null, null))

    assert(rord.gt(5, 7))
    assert(rord.lt(5, null))
    assert(rord.gt(null, 7))
  }
}

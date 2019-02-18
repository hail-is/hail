package is.hail.annotations

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.language.implicitConversions

class AnnotationsSuite extends TestNGSuite {
  @Test def testExtendedOrdering() {
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

package is.hail.annotations

import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.virtual._
import is.hail.testUtils._
import is.hail.utils._
import is.hail.{HailSuite, TestUtils}
import org.testng.annotations.Test

import scala.language.implicitConversions

/**
  * This testing suite evaluates the functionality of the [[is.hail.annotations]] package
  */
class AnnotationsSuite extends HailSuite {
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

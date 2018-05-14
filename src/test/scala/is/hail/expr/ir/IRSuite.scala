package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._

import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class IRSuite extends TestNGSuite {
  @Test def testNonstrictEQ() {
    assertEvalsTo(nonstrictEQ(NA(TInt32()), NA(TInt32())), true)
    assertEvalsTo(nonstrictEQ(I32(5), I32(5)), true)
    assertEvalsTo(nonstrictEQ(NA(TInt32()), I32(5)), false)
  }
}

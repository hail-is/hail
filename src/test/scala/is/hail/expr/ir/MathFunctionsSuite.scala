package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.utils._
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class MathFunctionsSuite extends TestNGSuite {

  @Test def basicUnirootFunction() {
    val ir = Uniroot("x",
      ApplyBinaryPrimOp(Add(), Ref("x", TFloat64()), F64(3)),
      F64(-6), F64(0))

    assertEvalsTo(ir, -3)
  }

  @Test def testUnirootWithExternalBinding() {
    val fn = ApplyBinaryPrimOp(Add(),
      Ref("x", TFloat64()),
      Ref("b", TFloat64()))
    val ir = Let("b", F64(3),
      Uniroot("x", fn, F64(-6), F64(0)))

    assertEvalsTo(ir, -3)
  }
}

package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.expr.ir.functions.ArrayFunctions
import is.hail.utils._
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class MathFunctionsSuite extends TestNGSuite {

  val tfloat = TFloat64()

  @Test def basicUnirootFunction() {
    val ir = Uniroot("x",
      ApplyBinaryPrimOp(Add(), Ref("x", tfloat), F64(3)),
      F64(-6), F64(0))

    assertEvalsTo(ir, -3.0)
  }

  @Test def unirootWithExternalBinding() {
    val fn = ApplyBinaryPrimOp(Add(),
      Ref("x", tfloat),
      Ref("b", tfloat))
    val ir = Let("b", F64(3),
      Uniroot("x", fn, F64(-6), F64(0)))

    assertEvalsTo(ir, -3.0)
  }

  @Test def unirootWithRegionManipulation() {
    def sum(array: IR): IR =
      ArrayFold(array, F64(0), "sum", "i", ApplyBinaryPrimOp(Add(), Ref("sum", tfloat), Ref("i", tfloat)))
    val fn = ApplyBinaryPrimOp(Add(),
      sum(MakeArray(Seq(Ref("x", tfloat), Ref("x", tfloat)), TArray(tfloat))),
      Ref("b", tfloat))
    val ir = Let("b", F64(6),
      Uniroot("x", fn, F64(-6), F64(0)))

    assertEvalsTo(ir, -3.0)
  }
}

package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
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
  
  @Test def rpois() {
    val res0 = eval(invoke("rpois", I32(5), F64(1)))
    assert(TArray(TFloat64()).typeCheck(res0))
    val res = res0.asInstanceOf[IndexedSeq[Double]]
    assert(res.length == 5)
    assert(res.forall(_ >= 0))
    assert(res.forall(x => x == x.floor))
  }
  
  @Test def isnan() {
    assertEvalsTo(invoke("isnan", F32(0)), false)
    assertEvalsTo(invoke("isnan", F32(Float.NaN)), true)
    
    assertEvalsTo(invoke("isnan", F64(0)), false)
    assertEvalsTo(invoke("isnan", F64(Double.NaN)), true)
  }
  
  @Test def sign() {
    assertEvalsTo(invoke("sign", I32(2)), 1)
    assertEvalsTo(invoke("sign", I32(0)), 0)
    assertEvalsTo(invoke("sign", I32(-2)), -1)
    
    assertEvalsTo(invoke("sign", I64(2)), 1l)
    assertEvalsTo(invoke("sign", I64(0)), 0l)
    assertEvalsTo(invoke("sign", I64(-2)), -1l)

    assertEvalsTo(invoke("sign", F32(2)), 1.0f)
    assertEvalsTo(invoke("sign", F32(0)), 0.0f)
    assertEvalsTo(invoke("sign", F32(-2)), -1.0f)
    
    assertEvalsTo(invoke("sign", F64(2)), 1.0)
    assertEvalsTo(invoke("sign", F64(0)), 0.0)
    assertEvalsTo(invoke("sign", F64(-2)), -1.0)

    assert(eval(invoke("sign", F64(Double.NaN))).asInstanceOf[Double].isNaN)
    assertEvalsTo(invoke("sign", F64(Double.PositiveInfinity)), 1.0)
    assertEvalsTo(invoke("sign", F64(Double.NegativeInfinity)), -1.0)    
  }

  @Test def entropy() {
    assertEvalsTo(invoke("entropy", Str("")), 0.0)
    assertEvalsTo(invoke("entropy", Str("a")), 0.0)
    assertEvalsTo(invoke("entropy", Str("aa")), 0.0)
    assertEvalsTo(invoke("entropy", Str("ac")), 1.0)
    assertEvalsTo(invoke("entropy", Str("accctg")), 1.7924812503605778)
  }
}

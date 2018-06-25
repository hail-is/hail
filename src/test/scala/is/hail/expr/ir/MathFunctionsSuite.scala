package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.utils._
import is.hail.TestUtils._
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}
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

  @Test def unirootIsStrictInMinAndMax() {
    assertEvalsTo(
      Uniroot("x", Ref("x", tfloat), F64(-6), NA(tfloat)),
      null)
    assertEvalsTo(
      Uniroot("x", Ref("x", tfloat), NA(tfloat), F64(0)),
      null)
  }

  @DataProvider(name = "chisq")
  def chisqData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, 0, Double.NaN, Double.NaN),
    Array(0, 1, 1, 1, 0.38647623077123266, 0.0),
    Array(1, 0, 1, 1, 0.38647623077123266, Double.PositiveInfinity),
    Array(1, 1, 0, 1, 0.38647623077123266, Double.PositiveInfinity),
    Array(1, 1, 1, 0, 0.38647623077123266, 0.0),
    Array(10, 10, 10, 10, 1.0, 1.0),
    Array(51, 43, 22, 92, 1.462626e-7, (51.0 * 92) / (22 * 43))
  )
  
  @Test(dataProvider = "chisq")
  def chisq(a: Int, b: Int, c: Int, d: Int, pValue: Double, oddsRatio: Double) {
      val r = eval(invoke("chisq", a, b, c, d)).asInstanceOf[Row]
      assert(D0_==(pValue, r.getDouble(0)))
      assert(D0_==(oddsRatio, r.getDouble(1)))
  }

  @DataProvider(name = "fet")
  def fetData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, 0, Double.NaN, Double.NaN, Double.NaN, Double.NaN),
    Array(10, 10, 10, 10, 1.0, 1.0, 0.243858, 4.100748),
    Array(51, 43, 22, 92, 2.1565e-7, 4.918058, 2.565937, 9.677930)
  )

  @Test(dataProvider = "fet")
  def fet(a: Int, b: Int, c: Int, d: Int, pValue: Double, oddsRatio: Double, confLower: Double, confUpper: Double) {
    val r = eval(invoke("fet", a, b, c, d)).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(oddsRatio, r.getDouble(1)))
    assert(D0_==(confLower, r.getDouble(2)))
    assert(D0_==(confUpper, r.getDouble(3)))
  }

  @DataProvider(name = "ctt")
  def cttData(): Array[Array[Any]] = Array(
    Array(51, 43, 22, 92, 22, 1.462626e-7, 4.95983087),
    Array(51, 43, 22, 92, 23, 2.1565e-7, 4.91805817)
  )
  
  @Test(dataProvider = "ctt")
  def ctt(a: Int, b: Int, c: Int, d: Int, minCellCount: Int, pValue: Double, oddsRatio: Double) {
    val r = eval(invoke("ctt", a, b, c, d, minCellCount)).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(oddsRatio, r.getDouble(1)))
  }

  @DataProvider(name = "hwe")
  def hweData(): Array[Array[Any]] = Array(
    Array(0, 0, 0, Double.NaN, 0.5),
    Array(1, 2, 1, 0.57142857, 0.65714285),
    Array(0, 1, 0, 1.0, 0.5),
    Array(100, 200, 100, 0.50062578, 0.96016808)
  )
  
  @Test(dataProvider = "hwe")
  def hwe(nHomRef: Int, nHet: Int, nHomVar: Int, pValue: Double, expectedHetFreq: Double) {
    val r = eval(invoke("hwe", nHomRef, nHet, nHomVar)).asInstanceOf[Row]
    assert(D0_==(pValue, r.getDouble(0)))
    assert(D0_==(expectedHetFreq, r.getDouble(1)))
  }
}

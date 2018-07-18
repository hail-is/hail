package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class GenotypeFunctionsSuite extends TestNGSuite {

  @DataProvider(name="gps")
  def gpData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(1.0, 0.0, 0.0), 0.0),
    Array(IndexedSeq(0.0, 1.0, 0.0), 1.0),
    Array(IndexedSeq(0.0, 0.0, 1.0), 2.0),
    Array(IndexedSeq(0.5, 0.5, 0.0), 0.5),
    Array(IndexedSeq(0.0, 0.5, 0.5), 1.5))

  @Test(dataProvider="gps")
  def testDosage(gp: IndexedSeq[java.lang.Double], expected: java.lang.Double) {
    assertEvalsTo(invoke("dosage", toIRDoubleArray(gp)), expected)
  }

  def testDosageLength() {
    assertFatal(invoke("dosage", IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }
}

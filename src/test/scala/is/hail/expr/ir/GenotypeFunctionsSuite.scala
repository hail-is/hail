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

  @DataProvider(name="pls")
  def plData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(0, 20, 100), 0.009900990296049406),
    Array(IndexedSeq(20, 0, 100), 0.9900990100009803),
    Array(IndexedSeq(20, 100, 0), 1.980198019704931))

  @Test(dataProvider="gps")
  def testDosage(gp: IndexedSeq[java.lang.Double], expected: java.lang.Double) {
    assertEvalsTo(invoke("dosage", toIRDoubleArray(gp)), expected)
  }

  @Test(dataProvider="pls")
  def testPLDosage(pl: IndexedSeq[Integer], expected: java.lang.Double) {
    assertEvalsTo(invoke("plDosage", toIRArray(pl)), expected)
  }

  def testDosageLength() {
    assertFatal(invoke("dosage", IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }

  def testPLDosageLength() {
    assertFatal(invoke("plDosage", IRArray(1, 2)), "length")
    assertFatal(invoke("plDosage", IRArray(1, 2, 0, 0)), "length")
  }
}

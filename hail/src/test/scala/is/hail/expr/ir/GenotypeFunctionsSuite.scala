package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types.virtual.TFloat64
import is.hail.utils.FastIndexedSeq
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class GenotypeFunctionsSuite extends TestNGSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name="gps")
  def gpData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(1.0, 0.0, 0.0), 0.0),
    Array(FastIndexedSeq(0.0, 1.0, 0.0), 1.0),
    Array(FastIndexedSeq(0.0, 0.0, 1.0), 2.0),
    Array(FastIndexedSeq(0.5, 0.5, 0.0), 0.5),
    Array(FastIndexedSeq(0.0, 0.5, 0.5), 1.5))

  @DataProvider(name="pls")
  def plData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(0, 20, 100), 0.009900990296049406),
    Array(FastIndexedSeq(20, 0, 100), 0.9900990100009803),
    Array(FastIndexedSeq(20, 100, 0), 1.980198019704931))

  @Test(dataProvider="gps")
  def testDosage(gp: IndexedSeq[java.lang.Double], expected: java.lang.Double) {
    assertEvalsTo(invoke("dosage", TFloat64, toIRDoubleArray(gp)), expected)
  }

  @Test(dataProvider="pls")
  def testPLDosage(pl: IndexedSeq[Integer], expected: java.lang.Double) {
    assertEvalsTo(invoke("plDosage", TFloat64, toIRArray(pl)), expected)
  }

  def testDosageLength() {
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }

  def testPLDosageLength() {
    assertFatal(invoke("plDosage", TFloat64, IRArray(1, 2)), "length")
    assertFatal(invoke("plDosage", TFloat64, IRArray(1, 2, 0, 0)), "length")
  }
}

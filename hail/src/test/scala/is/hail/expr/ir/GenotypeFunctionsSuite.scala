package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual.TFloat64
import is.hail.utils.FastSeq

import org.testng.annotations.{DataProvider, Test}

class GenotypeFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name = "gps")
  def gpData(): Array[Array[Any]] = Array(
    Array(FastSeq(1.0, 0.0, 0.0), 0.0),
    Array(FastSeq(0.0, 1.0, 0.0), 1.0),
    Array(FastSeq(0.0, 0.0, 1.0), 2.0),
    Array(FastSeq(0.5, 0.5, 0.0), 0.5),
    Array(FastSeq(0.0, 0.5, 0.5), 1.5),
    Array(null, null),
    Array(FastSeq(null, null, null), null),
    Array(FastSeq(null, 0.5, 0.5), 1.5),
    Array(FastSeq(0.0, null, 1.0), null),
    Array(FastSeq(0.0, 0.5, null), null),
  )

  @Test(dataProvider = "gps")
  def testDosage(gp: IndexedSeq[java.lang.Double], expected: java.lang.Double) {
    assertEvalsTo(invoke("dosage", TFloat64, toIRDoubleArray(gp)), expected)
  }

  @Test def testDosageLength() {
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }
}

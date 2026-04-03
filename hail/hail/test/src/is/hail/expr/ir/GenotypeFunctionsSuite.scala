package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual.TFloat64

class GenotypeFunctionsSuite extends HailSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  object checkDosage extends TestCases {
    def apply(
      gp: IndexedSeq[java.lang.Double],
      expected: java.lang.Double,
    )(implicit
      loc: munit.Location
    ): Unit = test("dosage") {
      assertEvalsTo(invoke("dosage", TFloat64, toIRDoubleArray(gp)), expected)
    }
  }

  checkDosage(FastSeq(1.0, 0.0, 0.0), 0.0)
  checkDosage(FastSeq(0.0, 1.0, 0.0), 1.0)
  checkDosage(FastSeq(0.0, 0.0, 1.0), 2.0)
  checkDosage(FastSeq(0.5, 0.5, 0.0), 0.5)
  checkDosage(FastSeq(0.0, 0.5, 0.5), 1.5)
  checkDosage(null, null)
  checkDosage(FastSeq(null, null, null), null)
  checkDosage(FastSeq(null, 0.5, 0.5), 1.5)
  checkDosage(FastSeq(0.0, null, 1.0), null)
  checkDosage(FastSeq(0.0, 0.5, null), null)

  test("dosageLength") {
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }
}

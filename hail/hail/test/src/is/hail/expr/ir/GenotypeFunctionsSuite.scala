package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual.TFloat64

import org.junit.jupiter.api.Test

class GenotypeFunctionsSuite {

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  def testDosage() = ArraySeq[(IndexedSeq[java.lang.Double], java.lang.Double)](
    (FastSeq(1.0, 0.0, 0.0), 0.0),
    (FastSeq(0.0, 1.0, 0.0), 1.0),
    (FastSeq(0.0, 0.0, 1.0), 2.0),
    (FastSeq(0.5, 0.5, 0.0), 0.5),
    (FastSeq(0.0, 0.5, 0.5), 1.5),
    (null, null),
    (FastSeq(null, null, null), null),
    (FastSeq(null, 0.5, 0.5), 1.5),
    (FastSeq(0.0, null, 1.0), null),
    (FastSeq(0.0, 0.5, null), null),
  )

  @ParameterizedTest
  def testDosage(
    gp: IndexedSeq[java.lang.Double],
    expected: java.lang.Double,
  )(implicit
    ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("dosage", TFloat64, toIRDoubleArray(gp)), expected)

  @Test def testDosageLength(implicit ctx: ExecuteContext): Unit = {
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5)), "length")
    assertFatal(invoke("dosage", TFloat64, IRDoubleArray(1.0, 1.5, 0.0, 0.0)), "length")
  }
}

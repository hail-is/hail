package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PIntervalSuite extends HailSuite {
  @Test def copyTests() {
    def runTests(forceDeep: Boolean) {
      PhysicalTestUtils.copyTestExecutor(PCanonicalInterval(PInt64()), PCanonicalInterval(PInt64()),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PCanonicalInterval(PInt64(true)), PCanonicalInterval(PInt64()),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PCanonicalInterval(PInt64()), PCanonicalInterval(PInt64(true)),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        expectCompileErr = true, forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PCanonicalInterval(PInt64(true)), PCanonicalInterval(PInt64(true)),
        Interval(IntervalEndpoint(1000L, 1), IntervalEndpoint(1000L, 1)),
        forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}

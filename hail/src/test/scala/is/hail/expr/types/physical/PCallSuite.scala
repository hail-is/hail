package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PCallSuite extends HailSuite {
  @Test def copyTests() {
    def runTests(forceDeep: Boolean) {
      PhysicalTestUtils.copyTestExecutor(PCanonicalCall(), PCanonicalCall(),
        2,
        forceDeep = forceDeep)

      // downcast at top level allowed, since address and therefore value must be present
      PhysicalTestUtils.copyTestExecutor(PCanonicalCall(), PCanonicalCall(true),
        2,
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PArray(PCanonicalCall(true), true), PArray(PCanonicalCall()),
        IndexedSeq(2, 3), forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PArray(PCanonicalCall(), true), PArray(PCanonicalCall(true)),
        IndexedSeq(2, 3), expectCompileErr = true, forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}

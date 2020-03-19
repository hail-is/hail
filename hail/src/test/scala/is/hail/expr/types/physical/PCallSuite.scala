package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PCallSuite extends HailSuite {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PCanonicalCall(), PCanonicalCall(),
        2,
        deepCopy = deepCopy, interpret = interpret)

      // downcast at top level allowed, since PCanonicalCall wraps a primitive
      PhysicalTestUtils.copyTestExecutor(PCanonicalCall(), PCanonicalCall(true),
        2,
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PArray(PCanonicalCall(true), true), PArray(PCanonicalCall()),
        IndexedSeq(2, 3), deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PArray(PCanonicalCall(), true), PArray(PCanonicalCall(true)),
        IndexedSeq(2, 3), expectCompileErr = true, deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}

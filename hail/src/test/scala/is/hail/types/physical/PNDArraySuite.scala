package is.hail.types.physical

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Region, ScalaToRegionValue}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.utils._
import org.testng.annotations.Test

class PNDArraySuite extends PhysicalTestUtils {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      copyTestExecutor(PCanonicalNDArray(PInt64(true), 1), PCanonicalNDArray(PInt64(true), 1), Annotation(Annotation(1L), Annotation(1L), IndexedSeq(4L,5L,6L)),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }
}

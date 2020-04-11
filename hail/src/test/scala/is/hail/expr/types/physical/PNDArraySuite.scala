package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PNDArraySuite extends HailSuite {
  @Test def copyTests() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PCanonicalNDArray(PInt64(true), 1), PCanonicalNDArray(PInt64(true), 1), Annotation(Annotation(1L), Annotation(1L), IndexedSeq(4L,5L,6L)),
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, true)
    runTests(false, true)
  }
}

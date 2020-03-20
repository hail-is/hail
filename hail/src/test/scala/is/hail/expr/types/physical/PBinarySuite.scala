package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PBinarySuite extends HailSuite {
  @Test def testCopy() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PString(), PString(), "",
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PString(), PString(true), "TopLevelDowncastAllowed",
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PString(true), PString(), "UpcastAllowed",
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}

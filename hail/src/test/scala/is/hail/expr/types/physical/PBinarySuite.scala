package is.hail.expr.types.physical

import is.hail.HailSuite
import org.testng.annotations.Test

class PBinarySuite extends HailSuite {
  @Test def testCopy() {
    def runTests(deepCopy: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PCanonicalString(), PCanonicalString(), "",
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PCanonicalString(), PCanonicalString(true), "TopLevelDowncastAllowed",
        deepCopy = deepCopy, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PCanonicalString(true), PCanonicalString(), "UpcastAllowed",
        deepCopy = deepCopy, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}

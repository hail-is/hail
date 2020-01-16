package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PBinarySuite extends HailSuite {
  @Test def testCopy() {
    def runTests(forceDeep: Boolean, interpret: Boolean = false) {
      PhysicalTestUtils.copyTestExecutor(PString(), PString(), "",
        forceDeep = forceDeep, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PString(), PString(true), "TopLevelDowncastAllowed",
        forceDeep = forceDeep, interpret = interpret)

      PhysicalTestUtils.copyTestExecutor(PString(true), PString(), "UpcastAllowed",
        forceDeep = forceDeep, interpret = interpret)
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}

package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PBinarySuite extends HailSuite {
  @Test def testCopy() {
    def runTests(forceDeep: Boolean) {
      PhysicalTestUtils.copyTestExecutor(PString(), PString(), "",
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PString(), PString(true), "TopLevelDowncastAllowed",
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PString(true), PString(), "UpcastAllowed",
        forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}

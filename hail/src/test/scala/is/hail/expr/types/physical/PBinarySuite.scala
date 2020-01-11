package is.hail.expr.types.physical

import is.hail.HailSuite
import is.hail.annotations.Annotation
import org.testng.annotations.Test

class PBinarySuite extends HailSuite {
  @Test def testCopy() {
    def runTests(forceDeep: Boolean) {
      PhysicalTestUtils.copyTestExecutor(PString(), PString(), "",
        forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PString(), PString(true), "DowncastDisabledByDefault",
        expectCompileErr = true, forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PString(), PString(true), "AllowDowncast",
        allowDowncast = true, forceDeep = forceDeep)

      // required because first stack entry in RVB must be non-null
      PhysicalTestUtils.copyTestExecutor(PArray(PString()), PArray(PString(true)), IndexedSeq("DowncastRuntimeFailOnNull", null),
        allowDowncast = true, expectRuntimeErr = true, forceDeep = forceDeep)

      PhysicalTestUtils.copyTestExecutor(PString(true), PString(), "Upcast",
        forceDeep = forceDeep)
    }

    runTests(true)
    runTests(false)
  }
}

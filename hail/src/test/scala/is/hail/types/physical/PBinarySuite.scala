package is.hail.types.physical

import org.testng.annotations.Test

class PBinarySuite extends PhysicalTestUtils {
  @Test def testCopy(): Unit = {
    def runTests(deepCopy: Boolean, interpret: Boolean = false): Unit = {
      copyTestExecutor(
        PCanonicalString(),
        PCanonicalString(),
        "",
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalString(),
        PCanonicalString(true),
        "TopLevelDowncastAllowed",
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalString(true),
        PCanonicalString(),
        "UpcastAllowed",
        deepCopy = deepCopy,
        interpret = interpret,
      )
    }

    runTests(true)
    runTests(false)

    runTests(true, interpret = true)
    runTests(false, interpret = true)
  }
}

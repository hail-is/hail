package is.hail.types.physical
import org.testng.annotations.Test

class PCallSuite extends PhysicalTestUtils {
  @Test def copyTests(): Unit = {
    def runTests(deepCopy: Boolean, interpret: Boolean = false): Unit = {
      copyTestExecutor(
        PCanonicalCall(),
        PCanonicalCall(),
        2,
        deepCopy = deepCopy,
        interpret = interpret,
      )

      // downcast at top level allowed, since PCanonicalCall wraps a primitive
      copyTestExecutor(
        PCanonicalCall(),
        PCanonicalCall(true),
        2,
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalArray(PCanonicalCall(true), true),
        PCanonicalArray(PCanonicalCall()),
        IndexedSeq(2, 3),
        deepCopy = deepCopy,
        interpret = interpret,
      )

      copyTestExecutor(
        PCanonicalArray(PCanonicalCall(), true),
        PCanonicalArray(PCanonicalCall(true)),
        IndexedSeq(2, 3),
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

package is.hail.expr.ir

import is.hail.utils._
import is.hail.TestUtils._
import is.hail.expr.types._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class StringSliceSuite extends TestNGSuite {
  @Test def zeroToLengthIsIdentity() {
    assertEvalsTo(StringSlice(Str("abc"), I32(0), I32(3)), "abc")
  }

  @Test def simpleSlicesMatchIntuition() {
    assertEvalsTo(StringSlice(Str("abc"), I32(3), I32(3)), "")
    assertEvalsTo(StringSlice(Str("abc"), I32(1), I32(3)), "bc")
    assertEvalsTo(StringSlice(Str("abc"), I32(2), I32(3)), "c")
    assertEvalsTo(StringSlice(Str("abc"), I32(0), I32(2)), "ab")
  }

  @Test def sizeZeroSliceIsEmptyString() {
    assertEvalsTo(StringSlice(Str("abc"), I32(2), I32(2)), "")
    assertEvalsTo(StringSlice(Str("abc"), I32(1), I32(1)), "")
    assertEvalsTo(StringSlice(Str("abc"), I32(0), I32(0)), "")
  }

  @Test def sliceMatchesJavaStringSubstring() {
    assertEvalsTo(
      StringSlice(Str("abc"), I32(0), I32(2)),
      "abc".substring(0, 2))
    assertEvalsTo(
      StringSlice(Str("foobarbaz"), I32(3), I32(5)),
      "foobarbaz".substring(3, 5))
  }

  @Test def isStrict() {
    assertEvalsTo(StringSlice(NA(TString()), I32(0), I32(2)), null)
    assertEvalsTo(StringSlice(NA(TString()), I32(-5), I32(-10)), null)
  }

  @Test def sliceCopyIsID() {
    assertEvalsTo(invoke("[:]", Str("abc")), "abc")
  }

  @Test def leftSliceMatchesIntuition() {
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(2)), "c")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(1)), "bc")
  }

  @Test def rightSliceMatchesIntuition() {
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(2)), "ab")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(1)), "a")
  }

  @Test def bothSideSliceMatchesIntuition() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(0), I32(2)), "ab")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(1), I32(3)), "bc")
  }

  @Test def leftSliceIsPythony() {
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(-1)), "c")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(-2)), "bc")
  }

  @Test def rightSliceIsPythony() {
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(-1)), "ab")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(-2)), "a")
  }

  @Test def sliceIsPythony() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-3), I32(-1)), "ab")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-3), I32(-2)), "a")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-2), I32(-1)), "b")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-2), I32(-2)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-3), I32(-3)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(1), I32(-1)), "b")
  }

  @Test def rawIROutOfBoundsFatals() {
    assertFatal(StringSlice(Str("abc"), I32(4), I32(4)),
      "string slice out of bounds or invalid: \"abc\"\\[4:4\\]")
    assertFatal(StringSlice(Str("abc"), I32(3), I32(2)),
      "string slice out of bounds or invalid: \"abc\"\\[3:2\\]")
    assertFatal(StringSlice(Str("abc"), I32(-1), I32(2)),
      "string slice out of bounds or invalid: \"abc\"\\[-1:2\\]")
    assertFatal(StringSlice(Str("abc"), I32(-1), I32(-1)),
      "string slice out of bounds or invalid: \"abc\"\\[-1:-1\\]")
    assertFatal(StringSlice(Str("abc"), I32(1), I32(-1)),
      "string slice out of bounds or invalid: \"abc\"\\[1:-1\\]")
  }

  @Test def bothSidesSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(4), I32(4)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(3), I32(2)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-1), I32(2)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-1), I32(-1)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(3), I32(3)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-10), I32(-5)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-5), I32(-10)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(-10), I32(-1)), "ab")
  }

  @Test def leftSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(15)), "")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(4)), "")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(3)), "")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(-3)), "abc")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(-4)), "abc")
    assertEvalsTo(invoke("[*:]", Str("abc"), I32(-100)), "abc")
  }

  @Test def rightSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(15)), "abc")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(4)), "abc")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(3)), "abc")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(-3)), "")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(-4)), "")
    assertEvalsTo(invoke("[:*]", Str("abc"), I32(-100)), "")
  }
}

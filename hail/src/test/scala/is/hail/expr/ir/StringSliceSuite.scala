package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.utils._
import is.hail.TestUtils._
import is.hail.expr.types._
import is.hail.expr.types.virtual.TString
import org.testng.annotations.Test

class StringSliceSuite extends HailSuite {
  implicit val execStrats = ExecStrategy.javaOnly

  @Test def unicodeSlicingSlicesCodePoints() {
    val poopEmoji = "\uD83D\uDCA9"
    val s = s"abc${ poopEmoji }def"

    // FIXME: The replacement character for slicing halfway into a
    // 2-codepoint-wide character differs between UTF8 and UTF16.
    // We've tested against the UTF8 character here since that's the encoding we
    // currently use, but the replacement character for utf16 is /ufffd.
    val replacementCharacter = "?"

    assertEvalsTo(invoke("[*:*]", Str(s), I32(0), I32(4)), s"abc$replacementCharacter")
    assertEvalsTo(invoke("[*:*]", Str(s), I32(4), I32(8)), s"${ replacementCharacter }def")
    assertEvalsTo(invoke("[*:*]", Str(s), I32(0), I32(5)), s"abc$poopEmoji")
  }

  @Test def zeroToLengthIsIdentity() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(0), I32(3)), "abc")
  }

  @Test def simpleSlicesMatchIntuition() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(3), I32(3)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(1), I32(3)), "bc")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(2), I32(3)), "c")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(0), I32(2)), "ab")
  }

  @Test def sizeZeroSliceIsEmptyString() {
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(2), I32(2)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(1), I32(1)), "")
    assertEvalsTo(invoke("[*:*]", Str("abc"), I32(0), I32(0)), "")
  }

  @Test def sliceMatchesJavaStringSubstring() {
    assertEvalsTo(
      invoke("[*:*]", Str("abc"), I32(0), I32(2)),
      "abc".substring(0, 2))
    assertEvalsTo(
      invoke("[*:*]", Str("foobarbaz"), I32(3), I32(5)),
      "foobarbaz".substring(3, 5))
  }

  @Test def isStrict() {
    assertEvalsTo(invoke("[*:*]", NA(TString()), I32(0), I32(2)), null)
    assertEvalsTo(invoke("[*:*]", NA(TString()), I32(-5), I32(-10)), null)
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

  @Test def testStringIndex() {
    assertEvalsTo(invoke("[]", In(0, TString()), I32(0)), FastIndexedSeq("Baz" -> TString()), "B")
    assertEvalsTo(invoke("[]", In(0, TString()), I32(1)), FastIndexedSeq("Baz" -> TString()), "a")
    assertEvalsTo(invoke("[]", In(0, TString()), I32(2)), FastIndexedSeq("Baz" -> TString()), "z")
    assertEvalsTo(invoke("[]", In(0, TString()), I32(-1)), FastIndexedSeq("Baz" -> TString()), "z")
    assertEvalsTo(invoke("[]", In(0, TString()), I32(-2)), FastIndexedSeq("Baz" -> TString()), "a")
    assertEvalsTo(invoke("[]", In(0, TString()), I32(-3)), FastIndexedSeq("Baz" -> TString()), "B")

    interceptFatal("string index out of bounds") {
      assertEvalsTo(invoke("[]", In(0, TString()), I32(3)), FastIndexedSeq("Baz" -> TString()), "B")
    }
    interceptFatal("string index out of bounds") {
      assertEvalsTo(invoke("[]", In(0, TString()), I32(-4)), FastIndexedSeq("Baz" -> TString()), "B")
    }
  }

  @Test def testStringCopy() {
    assertEvalsTo(invoke("[:]", In(0, TString())), FastIndexedSeq("Baz" -> TString()), "Baz")
  }
}

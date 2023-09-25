package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.types.virtual.TString
import is.hail.utils._
import is.hail.{ExecStrategy, HailSuite}
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

    assertEvalsTo(invoke("slice", TString, Str(s), I32(0), I32(4)), s"abc$replacementCharacter")
    assertEvalsTo(invoke("slice", TString, Str(s), I32(4), I32(8)), s"${ replacementCharacter }def")
    assertEvalsTo(invoke("slice", TString, Str(s), I32(0), I32(5)), s"abc$poopEmoji")
  }

  @Test def zeroToLengthIsIdentity() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(0), I32(3)), "abc")
  }

  @Test def simpleSlicesMatchIntuition() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(3), I32(3)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(1), I32(3)), "bc")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(2), I32(3)), "c")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(0), I32(2)), "ab")
  }

  @Test def sizeZeroSliceIsEmptyString() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(2), I32(2)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(1), I32(1)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(0), I32(0)), "")
  }

  @Test def substringMatchesJavaStringSubstring() {
    assertEvalsTo(
      invoke("substring", TString, Str("abc"), I32(0), I32(2)),
      "abc".substring(0, 2))
    assertEvalsTo(
      invoke("substring", TString, Str("foobarbaz"), I32(3), I32(5)),
      "foobarbaz".substring(3, 5))
  }

  @Test def isStrict() {
    assertEvalsTo(invoke("slice", TString, NA(TString), I32(0), I32(2)), null)
    assertEvalsTo(invoke("slice", TString, NA(TString), I32(-5), I32(-10)), null)
  }

  @Test def leftSliceMatchesIntuition() {
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(2)), "c")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(1)), "bc")
  }

  @Test def rightSliceMatchesIntuition() {
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(2)), "ab")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(1)), "a")
  }

  @Test def bothSideSliceMatchesIntuition() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(0), I32(2)), "ab")
    // assertEvalsTo(invoke("slice", TString, Str("abc"), I32(1), I32(3)), "bc")
  }

  @Test def leftSliceIsPythony() {
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(-1)), "c")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(-2)), "bc")
  }

  @Test def rightSliceIsPythony() {
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(-1)), "ab")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(-2)), "a")
  }

  @Test def sliceIsPythony() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-3), I32(-1)), "ab")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-3), I32(-2)), "a")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-2), I32(-1)), "b")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-2), I32(-2)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-3), I32(-3)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(1), I32(-1)), "b")
  }

  @Test def bothSidesSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(4), I32(4)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(3), I32(2)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-1), I32(2)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-1), I32(-1)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(3), I32(3)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-10), I32(-5)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-5), I32(-10)), "")
    assertEvalsTo(invoke("slice", TString, Str("abc"), I32(-10), I32(-1)), "ab")
  }

  @Test def leftSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(15)), "")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(4)), "")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(3)), "")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(-3)), "abc")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(-4)), "abc")
    assertEvalsTo(invoke("sliceRight", TString, Str("abc"), I32(-100)), "abc")
  }

  @Test def rightSliceFunctionOutOfBoundsNotFatal() {
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(15)), "abc")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(4)), "abc")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(3)), "abc")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(-3)), "")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(-4)), "")
    assertEvalsTo(invoke("sliceLeft", TString, Str("abc"), I32(-100)), "")
  }

  @Test def testStringIndex() {
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(0)), FastSeq("Baz" -> TString), "B")
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(1)), FastSeq("Baz" -> TString), "a")
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(2)), FastSeq("Baz" -> TString), "z")
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(-1)), FastSeq("Baz" -> TString), "z")
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(-2)), FastSeq("Baz" -> TString), "a")
    assertEvalsTo(invoke("index", TString, In(0, TString), I32(-3)), FastSeq("Baz" -> TString), "B")

    interceptFatal("string index out of bounds") {
      assertEvalsTo(invoke("index", TString, In(0, TString), I32(3)), FastSeq("Baz" -> TString), "B")
    }
    interceptFatal("string index out of bounds") {
      assertEvalsTo(invoke("index", TString, In(0, TString), I32(-4)), FastSeq("Baz" -> TString), "B")
    }
  }
}

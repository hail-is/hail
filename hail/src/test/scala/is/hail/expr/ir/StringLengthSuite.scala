package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.types._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class StringLengthSuite extends TestNGSuite {
  @Test def sameAsJavaStringLength() {
    assertEvalsTo(StringLength(Str("abc")), 3)
    assertEvalsTo(StringLength(Str("")), 0)
  }

  @Test def poopEmojiIsLengthFour() {
    // poop is wayyy out of the basic multilingual plane (BMP), so its encoded
    // with two UTF-16 code points. Behold, poop hex:
    //
    // NB: hail treats strings as arrays of 8 bit characters, so the poop emoji
    // has length 4
    val poopEmoji = new String(Array[Char](0xD83D, 0xDCA9))
    assertEvalsTo(StringLength(Str(poopEmoji)), 4)
    assertEvalsTo(StringLength(Str(poopEmoji + poopEmoji)), 8)
  }

  @Test def isStrict() {
    assertEvalsTo(StringLength(NA(TString())), null)
  }

  @Test def funRegistryLenWorks() {
    assertEvalsTo(invoke("len", NA(TString())), null)
    assertEvalsTo(invoke("len", Str("abc")), 3)
  }
}

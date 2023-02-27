package is.hail.types.virtual

import is.hail.HailSuite
import org.testng.Assert.{assertFalse, assertTrue}
import org.testng.annotations.Test

class TStructSuite extends HailSuite {
  @Test
  def testEmptyIsSubsequenceEmpty(): Unit =
    assertTrue(TStruct().isSubsequence(TStruct()), "empty")

  @Test
  def testEmptyIsSubsequenceOfNonEmpty(): Unit =
    assertTrue(TStruct().isSubsequence(TStruct("any" -> TVoid)))

  @Test
  def testIsSubsequenceOfItself(): Unit = {
    val typ = TStruct("any" -> TVoid)
    assertTrue(typ.isSubsequence(typ))
  }

  @Test
  def testIsNotSubsequenceWhenLonger(): Unit =
    assertFalse(TStruct("a" -> TInt32, "b" -> TString).isSubsequence(TStruct("a" -> TInt32)))
}

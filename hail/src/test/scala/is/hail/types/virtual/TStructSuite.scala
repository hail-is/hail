package is.hail.types.virtual

import is.hail.HailSuite
import org.testng.Assert.{assertFalse, assertTrue}
import org.testng.annotations.Test

class TStructSuite extends HailSuite {
  @Test
  def testEmptyIsPrefixEmpty(): Unit =
    assertTrue(TStruct().isPrefix(TStruct()), "empty")

  @Test
  def testEmptyIsPrefixOfNonEmpty(): Unit =
    assertTrue(TStruct().isPrefix(TStruct("any" -> TVoid)))

  @Test
  def testIsPrefixOfItself(): Unit = {
    val typ = TStruct("any" -> TVoid)
    assertTrue(typ.isPrefix(typ))
  }

  @Test
  def testIsNotPrefixWhenLonger(): Unit =
    assertFalse(TStruct("a" -> TInt32, "b" -> TString).isPrefix(TStruct("a" -> TInt32)))

  @Test
  def testIsPrefix(): Unit =
    assertTrue(TStruct("a" -> TInt32).isPrefix(TStruct("a" -> TInt32, "b" -> TString)))
}

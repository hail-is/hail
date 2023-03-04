package is.hail.types.virtual

import is.hail.HailSuite
import org.testng.Assert.{assertFalse, assertTrue}
import org.testng.annotations.{DataProvider, Test}

class TStructSuite extends HailSuite {
  @DataProvider(name = "isPrefixOf")
  def isPrefixOfData: Array[Array[Any]] =
    Array(
      Array(TStruct.empty, TStruct.empty, true),
      Array(TStruct.empty, TStruct("a" -> TVoid), true),
      Array(TStruct("a" -> TVoid), TStruct.empty, false),
      Array(TStruct("a" -> TVoid), TStruct("a" -> TVoid), true),
      Array(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
      Array(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      // isPrefixOf ignores field names and compares the ordered sequence of types.
      // Consider joins for example - we only care that the key fields have the same types
      // so we compare the key types (which are structs) for equality ignoring field names.
      // isPrefixOf is used in similar cases involving key types where we don't care about names.
      Array(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)
    )

  @Test(dataProvider = "isPrefixOf")
  def testIsPrefixOf(a: TStruct, b: TStruct, isPrefix: Boolean): Unit =
    assert(a.isPrefixOf(b) == isPrefix, s"expected $a `isPrefixOf` $b == $isPrefix")

  @DataProvider(name = "isSubsetOf")
  def isSubsetOfData: Array[Array[Any]] =
    Array(
      Array(TStruct.empty, TStruct.empty, true),
      Array(TStruct.empty, TStruct("a" -> TVoid), true),
      Array(TStruct("a" -> TVoid), TStruct.empty, false),
      Array(TStruct("a" -> TVoid), TStruct("a" -> TVoid), true),
      Array(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
      Array(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      Array(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)
    )

  @Test(dataProvider = "isSubsetOf")
  def testIsSubsetOf(a: TStruct, b: TStruct, isSubset: Boolean): Unit =
    assert(a.isSubsetOf(b) == isSubset, s"expected $a `isSubsetOf` $b == $isSubset")

}

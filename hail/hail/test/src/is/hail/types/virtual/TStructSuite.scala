package is.hail.types.virtual

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Inserter}
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row
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
      Array(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
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
      Array(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
    )

  @Test(dataProvider = "isSubsetOf")
  def testIsSubsetOf(a: TStruct, b: TStruct, isSubset: Boolean): Unit =
    assert(a.isSubsetOf(b) == isSubset, s"expected $a `isSubsetOf` $b == $isSubset")

  @DataProvider(name = "structInsert")
  def structInsertData: Array[Array[Any]] =
    Array(
      Array(TStruct("a" -> TInt32), FastSeq("a"), TInt32, TStruct("a" -> TInt32)),
      Array(TStruct("a" -> TInt32), FastSeq("b"), TInt32, TStruct("a" -> TInt32, "b" -> TInt32)),
      Array(TStruct("a" -> TInt32), FastSeq("a"), TVoid, TStruct("a" -> TVoid)),
      Array(
        TStruct("a" -> TInt32),
        FastSeq("a", "b"),
        TInt32,
        TStruct("a" -> TStruct("b" -> TInt32)),
      ),
      Array(TStruct.empty, FastSeq("a"), TInt32, TStruct("a" -> TInt32)),
      Array(TStruct.empty, FastSeq("a", "b"), TInt32, TStruct("a" -> TStruct("b" -> TInt32))),
    )

  @Test(dataProvider = "structInsert")
  def testStructInsert(base: TStruct, path: IndexedSeq[String], signature: Type, expected: TStruct)
    : Unit =
    assert(base.structInsert(signature, path) == expected)

  @Test def testInsertEmptyPath(): Unit =
    intercept[IllegalArgumentException] {
      TStruct.empty.insert(TInt32, FastSeq())
    }

  @DataProvider(name = "inserter")
  def inserterData: Array[Array[Any]] =
    Array(
      Array(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, null, 0, Row(0)),
      Array(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, Row(0), 1, Row(1)),
      Array(TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, null, 0, Row(null, 0)),
      Array(TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, Row(0), 1, Row(0, 1)),
      Array(TStruct.empty.insert(TInt32, FastSeq("a", "b"))._2, null, 0, Row(Row(0))),
      Array(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a", "b"))._2, Row(0), 1, Row(Row(1))),
    )

  @Test(dataProvider = "inserter")
  def testInsert(inserter: Inserter, base: Annotation, value: Any, expected: Annotation): Unit =
    assert(inserter(base, value) == expected)

  @DataProvider(name = "isIsomorphicTo")
  def isIsomorphicToData: Array[Array[Any]] =
    Array(
      Array(TStruct.empty, TStruct.empty, true),
      Array(TStruct.empty, TStruct("a" -> TVoid), false),
      Array(TStruct("a" -> TVoid), TStruct.empty, false),
      Array(TStruct("a" -> TVoid), TStruct("b" -> TVoid), true),
      Array(TStruct("a" -> TStruct("b" -> TVoid)), TStruct("b" -> TStruct("a" -> TVoid)), true),
      Array(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false),
      Array(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
    )

  @Test(dataProvider = "isIsomorphicTo")
  def testIsIsomorphicTo(a: TStruct, b: TStruct, isIsomorphic: Boolean): Unit =
    assert((a isIsomorphicTo b) == isIsomorphic, s"expected $a isIsomorphicTo $b == $isIsomorphic")

  @DataProvider(name = "isJoinableWith")
  def isJoinableWithData: Array[Array[Any]] =
    Array(
      Array(TStruct.empty, TStruct.empty, true),
      Array(TStruct.empty, TStruct("a" -> TVoid), false),
      Array(TStruct("a" -> TVoid), TStruct.empty, false),
      Array(TStruct("a" -> TVoid), TStruct("b" -> TVoid), true),
      Array(TStruct("a" -> TStruct("a" -> TVoid)), TStruct("b" -> TStruct("a" -> TVoid)), true),
      Array(TStruct("a" -> TStruct("a" -> TVoid)), TStruct("b" -> TStruct("b" -> TVoid)), false),
      Array(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      Array(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false),
    )

  @Test(dataProvider = "isJoinableWith")
  def testIsJoinableWith(a: TStruct, b: TStruct, isJoinable: Boolean): Unit =
    assert((a isJoinableWith b) == isJoinable, s"expected $a isJoinableWith $b == $isJoinable")
}

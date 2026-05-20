package is.hail.types.virtual

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.annotations.{Annotation, Inserter}
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class TStructSuite {
  def testIsPrefixOf() =
    ArraySeq[(TStruct, TStruct, Boolean)](
      (TStruct.empty, TStruct.empty, true),
      (TStruct.empty, TStruct("a" -> TVoid), true),
      (TStruct("a" -> TVoid), TStruct.empty, false),
      (TStruct("a" -> TVoid), TStruct("a" -> TVoid), true),
      (TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
      (TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      // isPrefixOf ignores field names and compares the ordered sequence of types.
      // Consider joins for example - we only care that the key fields have the same types
      // so we compare the key types (which are structs) for equality ignoring field names.
      // isPrefixOf is used in similar cases involving key types where we don't care about names.
      (TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
    )

  @ParameterizedTest
  def testIsPrefixOf(a: TStruct, b: TStruct, isPrefix: Boolean): Unit =
    assert(a.isPrefixOf(b) == isPrefix, s"expected $a `isPrefixOf` $b == $isPrefix")

  def testIsSubsetOf() =
    ArraySeq[(TStruct, TStruct, Boolean)](
      (TStruct.empty, TStruct.empty, true),
      (TStruct.empty, TStruct("a" -> TVoid), true),
      (TStruct("a" -> TVoid), TStruct.empty, false),
      (TStruct("a" -> TVoid), TStruct("a" -> TVoid), true),
      (TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
      (TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      (TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true),
    )

  @ParameterizedTest
  def testIsSubsetOf(a: TStruct, b: TStruct, isSubset: Boolean): Unit =
    assert(a.isSubsetOf(b) == isSubset, s"expected $a `isSubsetOf` $b == $isSubset")

  def testStructInsert() =
    ArraySeq[(TStruct, IndexedSeq[String], Type, TStruct)](
      (TStruct("a" -> TInt32), FastSeq("a"), TInt32, TStruct("a" -> TInt32)),
      (TStruct("a" -> TInt32), FastSeq("b"), TInt32, TStruct("a" -> TInt32, "b" -> TInt32)),
      (TStruct("a" -> TInt32), FastSeq("a"), TVoid, TStruct("a" -> TVoid)),
      (
        TStruct("a" -> TInt32),
        FastSeq("a", "b"),
        TInt32,
        TStruct("a" -> TStruct("b" -> TInt32)),
      ),
      (TStruct.empty, FastSeq("a"), TInt32, TStruct("a" -> TInt32)),
      (TStruct.empty, FastSeq("a", "b"), TInt32, TStruct("a" -> TStruct("b" -> TInt32))),
    )

  @ParameterizedTest
  def testStructInsert(base: TStruct, path: IndexedSeq[String], signature: Type, expected: TStruct)
    : Unit =
    assert(base.structInsert(signature, path) == expected)

  @Test def testInsertEmptyPath(): Unit =
    intercept[IllegalArgumentException] {
      TStruct.empty.insert(TInt32, FastSeq())
    }: Unit

  def testInsert() =
    ArraySeq[(Inserter, Annotation, Any, Annotation)](
      (TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, null, 0, Row(0)),
      (TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, Row(0), 1, Row(1)),
      (TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, null, 0, Row(null, 0)),
      (TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, Row(0), 1, Row(0, 1)),
      (TStruct.empty.insert(TInt32, FastSeq("a", "b"))._2, null, 0, Row(Row(0))),
      (TStruct("a" -> TInt32).insert(TInt32, FastSeq("a", "b"))._2, Row(0), 1, Row(Row(1))),
    )

  @ParameterizedTest
  def testInsert(inserter: Inserter, base: Annotation, value: Any, expected: Annotation): Unit =
    assert(inserter(base, value) == expected)

  def testIsIsomorphicTo() =
    ArraySeq[(TStruct, TStruct, Boolean)](
      (TStruct.empty, TStruct.empty, true),
      (TStruct.empty, TStruct("a" -> TVoid), false),
      (TStruct("a" -> TVoid), TStruct.empty, false),
      (TStruct("a" -> TVoid), TStruct("b" -> TVoid), true),
      (TStruct("a" -> TStruct("b" -> TVoid)), TStruct("b" -> TStruct("a" -> TVoid)), true),
      (TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false),
      (TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
    )

  @ParameterizedTest
  def testIsIsomorphicTo(a: TStruct, b: TStruct, isIsomorphic: Boolean): Unit =
    assert((a isIsomorphicTo b) == isIsomorphic, s"expected $a isIsomorphicTo $b == $isIsomorphic")

  def testIsJoinableWith() =
    ArraySeq[(TStruct, TStruct, Boolean)](
      (TStruct.empty, TStruct.empty, true),
      (TStruct.empty, TStruct("a" -> TVoid), false),
      (TStruct("a" -> TVoid), TStruct.empty, false),
      (TStruct("a" -> TVoid), TStruct("b" -> TVoid), true),
      (TStruct("a" -> TStruct("a" -> TVoid)), TStruct("b" -> TStruct("a" -> TVoid)), true),
      (TStruct("a" -> TStruct("a" -> TVoid)), TStruct("b" -> TStruct("b" -> TVoid)), false),
      (TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false),
      (TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false),
    )

  @ParameterizedTest
  def testIsJoinableWith(a: TStruct, b: TStruct, isJoinable: Boolean): Unit =
    assert((a isJoinableWith b) == isJoinable, s"expected $a isJoinableWith $b == $isJoinable")
}

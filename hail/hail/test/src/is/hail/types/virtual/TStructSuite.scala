package is.hail.types.virtual

import is.hail.HailSuite
import is.hail.annotations.{Annotation, Inserter}
import is.hail.collection.FastSeq

import org.apache.spark.sql.Row

class TStructSuite extends HailSuite {
  object checkIsPrefixOf extends TestCases {
    def apply(
      a: TStruct,
      b: TStruct,
      isPrefix: Boolean,
    )(implicit loc: munit.Location
    ): Unit = test("isPrefixOf") {
      assert(a.isPrefixOf(b) == isPrefix, s"expected $a `isPrefixOf` $b == $isPrefix")
    }
  }

  checkIsPrefixOf(TStruct.empty, TStruct.empty, true)
  checkIsPrefixOf(TStruct.empty, TStruct("a" -> TVoid), true)
  checkIsPrefixOf(TStruct("a" -> TVoid), TStruct.empty, false)
  checkIsPrefixOf(TStruct("a" -> TVoid), TStruct("a" -> TVoid), true)
  checkIsPrefixOf(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)
  checkIsPrefixOf(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false)
  // isPrefixOf ignores field names and compares the ordered sequence of types.
  // Consider joins for example - we only care that the key fields have the same types
  // so we compare the key types (which are structs) for equality ignoring field names.
  // isPrefixOf is used in similar cases involving key types where we don't care about names.
  checkIsPrefixOf(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)

  object checkIsSubsetOf extends TestCases {
    def apply(
      a: TStruct,
      b: TStruct,
      isSubset: Boolean,
    )(implicit loc: munit.Location
    ): Unit = test("isSubsetOf") {
      assert(a.isSubsetOf(b) == isSubset, s"expected $a `isSubsetOf` $b == $isSubset")
    }
  }

  checkIsSubsetOf(TStruct.empty, TStruct.empty, true)
  checkIsSubsetOf(TStruct.empty, TStruct("a" -> TVoid), true)
  checkIsSubsetOf(TStruct("a" -> TVoid), TStruct.empty, false)
  checkIsSubsetOf(TStruct("a" -> TVoid), TStruct("a" -> TVoid), true)
  checkIsSubsetOf(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)
  checkIsSubsetOf(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false)
  checkIsSubsetOf(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), true)

  object checkStructInsert extends TestCases {
    def apply(
      base: TStruct,
      path: IndexedSeq[String],
      signature: Type,
      expected: TStruct,
    )(implicit loc: munit.Location
    ): Unit = test("structInsert") {
      assertEquals(base.structInsert(signature, path), expected)
    }
  }

  checkStructInsert(TStruct("a" -> TInt32), FastSeq("a"), TInt32, TStruct("a" -> TInt32))

  checkStructInsert(
    TStruct("a" -> TInt32),
    FastSeq("b"),
    TInt32,
    TStruct("a" -> TInt32, "b" -> TInt32),
  )

  checkStructInsert(TStruct("a" -> TInt32), FastSeq("a"), TVoid, TStruct("a" -> TVoid))

  checkStructInsert(
    TStruct("a" -> TInt32),
    FastSeq("a", "b"),
    TInt32,
    TStruct("a" -> TStruct("b" -> TInt32)),
  )

  checkStructInsert(TStruct.empty, FastSeq("a"), TInt32, TStruct("a" -> TInt32))

  checkStructInsert(
    TStruct.empty,
    FastSeq("a", "b"),
    TInt32,
    TStruct("a" -> TStruct("b" -> TInt32)),
  )

  test("InsertEmptyPath") {
    intercept[IllegalArgumentException] {
      TStruct.empty.insert(TInt32, FastSeq())
    }
  }

  object checkInsert extends TestCases {
    def apply(
      inserter: Inserter,
      base: Annotation,
      value: Any,
      expected: Annotation,
    )(implicit loc: munit.Location
    ): Unit = test("insert") {
      assertEquals(inserter(base, value), expected)
    }
  }

  checkInsert(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, null, 0, Row(0))
  checkInsert(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a"))._2, Row(0), 1, Row(1))
  checkInsert(TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, null, 0, Row(null, 0))
  checkInsert(TStruct("a" -> TInt32).insert(TInt32, FastSeq("b"))._2, Row(0), 1, Row(0, 1))
  checkInsert(TStruct.empty.insert(TInt32, FastSeq("a", "b"))._2, null, 0, Row(Row(0)))
  checkInsert(TStruct("a" -> TInt32).insert(TInt32, FastSeq("a", "b"))._2, Row(0), 1, Row(Row(1)))

  object checkIsIsomorphicTo extends TestCases {
    def apply(
      a: TStruct,
      b: TStruct,
      isIsomorphic: Boolean,
    )(implicit loc: munit.Location
    ): Unit = test("isIsomorphicTo") {
      assert(
        (a isIsomorphicTo b) == isIsomorphic,
        s"expected $a isIsomorphicTo $b == $isIsomorphic",
      )
    }
  }

  checkIsIsomorphicTo(TStruct.empty, TStruct.empty, true)
  checkIsIsomorphicTo(TStruct.empty, TStruct("a" -> TVoid), false)
  checkIsIsomorphicTo(TStruct("a" -> TVoid), TStruct.empty, false)
  checkIsIsomorphicTo(TStruct("a" -> TVoid), TStruct("b" -> TVoid), true)

  checkIsIsomorphicTo(
    TStruct("a" -> TStruct("b" -> TVoid)),
    TStruct("b" -> TStruct("a" -> TVoid)),
    true,
  )

  checkIsIsomorphicTo(TStruct("a" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false)
  checkIsIsomorphicTo(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false)

  object checkIsJoinableWith extends TestCases {
    def apply(
      a: TStruct,
      b: TStruct,
      isJoinable: Boolean,
    )(implicit loc: munit.Location
    ): Unit = test("isJoinableWith") {
      assert((a isJoinableWith b) == isJoinable, s"expected $a isJoinableWith $b == $isJoinable")
    }
  }

  checkIsJoinableWith(TStruct.empty, TStruct.empty, true)
  checkIsJoinableWith(TStruct.empty, TStruct("a" -> TVoid), false)
  checkIsJoinableWith(TStruct("a" -> TVoid), TStruct.empty, false)
  checkIsJoinableWith(TStruct("a" -> TVoid), TStruct("b" -> TVoid), true)

  checkIsJoinableWith(
    TStruct("a" -> TStruct("a" -> TVoid)),
    TStruct("b" -> TStruct("a" -> TVoid)),
    true,
  )

  checkIsJoinableWith(
    TStruct("a" -> TStruct("a" -> TVoid)),
    TStruct("b" -> TStruct("b" -> TVoid)),
    false,
  )

  checkIsJoinableWith(TStruct("a" -> TVoid, "b" -> TVoid), TStruct("a" -> TVoid), false)
  checkIsJoinableWith(TStruct("b" -> TVoid), TStruct("a" -> TVoid, "b" -> TVoid), false)
}

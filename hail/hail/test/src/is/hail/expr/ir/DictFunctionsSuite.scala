package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.collection.FastSeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{NA, ToSet, ToStream}
import is.hail.types.virtual._

import org.apache.spark.sql.Row

class DictFunctionsSuite extends HailSuite {
  def tuplesToMap(a: IndexedSeq[(Integer, Integer)]): Map[Integer, Integer] =
    Option(a).map(_.filter(_ != null).toMap).orNull

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  val basicData: Array[IndexedSeq[(Integer, Integer)]] = Array(
    IndexedSeq((1, 3), (2, 7)),
    IndexedSeq((1, 3), (2, null), null, (null, 1), (3, 7)),
    IndexedSeq(),
    IndexedSeq(null),
    null,
  )

  object checkDictFromArray extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)]
    )(implicit loc: munit.Location
    ): Unit = test("dictFromArray") {
      assertEvalsTo(invoke("dict", TDict(TInt32, TInt32), toIRPairArray(a)), tuplesToMap(a))
      assertEvalsTo(toIRDict(a), tuplesToMap(a))
    }
  }

  basicData.foreach(checkDictFromArray(_))

  object checkDictFromSet extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)]
    )(implicit loc: munit.Location
    ): Unit = test("dictFromSet") {
      assertEvalsTo(
        invoke("dict", TDict(TInt32, TInt32), ToSet(ToStream(toIRPairArray(a)))),
        tuplesToMap(a),
      )
    }
  }

  basicData.foreach(checkDictFromSet(_))

  object checkIsEmpty extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)]
    )(implicit loc: munit.Location
    ): Unit = test("isEmpty") {
      assertEvalsTo(
        invoke("isEmpty", TBoolean, toIRDict(a)),
        Option(a).map(_.forall(_ == null)).orNull,
      )
    }
  }

  basicData.foreach(checkIsEmpty(_))

  val dictToArrayData: Array[(IndexedSeq[(Integer, Integer)], IndexedSeq[Row])] = Array(
    (FastSeq[(Integer, Integer)]((1, 3), (2, 7)), FastSeq(Row(1, 3), Row(2, 7))),
    (
      FastSeq[(Integer, Integer)]((1, 3), (2, null), null, (null, 1), (3, 7)),
      FastSeq(Row(1, 3), Row(2, null), Row(3, 7), Row(null, 1)),
    ),
    (FastSeq[(Integer, Integer)](), FastSeq()),
    (FastSeq[(Integer, Integer)](null), FastSeq()),
    (null, null),
  )

  object checkDictToArray extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)],
      expected: IndexedSeq[Row],
    )(implicit loc: munit.Location
    ): Unit = test("dictToArray") {
      implicit val execStrats = Set(ExecStrategy.JvmCompile)
      assertEvalsTo(invoke("dictToArray", TArray(TTuple(TInt32, TInt32)), toIRDict(a)), expected)
    }
  }

  dictToArrayData.foreach { case (a, expected) => checkDictToArray(a, expected) }

  val keysAndValuesData
    : Array[(IndexedSeq[(Integer, Integer)], IndexedSeq[Integer], IndexedSeq[Integer])] = Array(
    (FastSeq[(Integer, Integer)]((1, 3), (2, 7)), FastSeq[Integer](1, 2), FastSeq[Integer](3, 7)),
    (
      FastSeq[(Integer, Integer)]((1, 3), (2, null), null, (null, 1), (3, 7)),
      FastSeq[Integer](1, 2, 3, null),
      FastSeq[Integer](3, null, 7, 1),
    ),
    (FastSeq[(Integer, Integer)](), FastSeq[Integer](), FastSeq[Integer]()),
    (FastSeq[(Integer, Integer)](null), FastSeq[Integer](), FastSeq[Integer]()),
    (null, null, null),
  )

  object checkKeySet extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)],
      keys: IndexedSeq[Integer],
      values: IndexedSeq[Integer],
    )(implicit loc: munit.Location
    ): Unit = test("keySet") {
      assertEvalsTo(invoke("keySet", TSet(TInt32), toIRDict(a)), Option(keys).map(_.toSet).orNull)
    }
  }

  keysAndValuesData.foreach { case (a, keys, values) => checkKeySet(a, keys, values) }

  object checkKeys extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)],
      keys: IndexedSeq[Integer],
      values: IndexedSeq[Integer],
    )(implicit loc: munit.Location
    ): Unit = test("keys") {
      assertEvalsTo(invoke("keys", TArray(TInt32), toIRDict(a)), keys)
    }
  }

  keysAndValuesData.foreach { case (a, keys, values) => checkKeys(a, keys, values) }

  object checkValues extends TestCases {
    def apply(
      a: IndexedSeq[(Integer, Integer)],
      keys: IndexedSeq[Integer],
      values: IndexedSeq[Integer],
    )(implicit loc: munit.Location
    ): Unit = test("values") {
      assertEvalsTo(invoke("values", TArray(TInt32), toIRDict(a)), values)
    }
  }

  keysAndValuesData.foreach { case (a, keys, values) => checkValues(a, keys, values) }

  val d = IRDict((1, 3), (3, 7), (5, null), (null, 5))
  val dwoutna = IRDict((1, 3), (3, 7), (5, null))
  val na = NA(TInt32)

  test("dictGet") {
    assertEvalsTo(invoke("get", TInt32, NA(TDict(TInt32, TInt32)), 1, na), null)
    assertEvalsTo(invoke("get", TInt32, d, 0, na), null)
    assertEvalsTo(invoke("get", TInt32, d, 1, na), 3)
    assertEvalsTo(invoke("get", TInt32, d, 2, na), null)
    assertEvalsTo(invoke("get", TInt32, d, 3, 50), 7)
    assertEvalsTo(invoke("get", TInt32, d, 4, -7), -7)
    assertEvalsTo(invoke("get", TInt32, d, 5, 50), null)
    assertEvalsTo(invoke("get", TInt32, d, na, 50), 5)
    assertEvalsTo(invoke("get", TInt32, dwoutna, na, 50), 50)
    assertEvalsTo(invoke("get", TInt32, d, 100, 50), 50)
    assertEvalsTo(invoke("get", TInt32, d, 100, na), null)
    assertEvalsTo(invoke("get", TInt32, dwoutna, 100, 50), 50)

    assertEvalsTo(invoke("get", TInt32, IRDict(), 100, na), null)
    assertEvalsTo(invoke("get", TInt32, IRDict(), 100, 50), 50)

    assertEvalsTo(invoke("index", TInt32, d, 1), 3)
    assertEvalsTo(invoke("index", TInt32, d, 5), null)
    assertEvalsTo(invoke("index", TInt32, d, na), 5)

    assertFatal(invoke("index", TInt32, d, -5), "dictionary")
    assertFatal(invoke("index", TInt32, d, 100), "dictionary")
    assertFatal(invoke("index", TInt32, IRDict(), 100), "dictionary")
  }

  test("dictContains") {
    assertEvalsTo(invoke("contains", TBoolean, d, 0), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 1), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 2), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 3), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 4), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 5), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 100), false)
    assertEvalsTo(invoke("contains", TBoolean, dwoutna, 100), false)
    assertEvalsTo(invoke("contains", TBoolean, d, na), true)

    assert(eval(invoke("contains", TBoolean, IRDict(), 100)) == false)
    assertEvalsTo(invoke("contains", TBoolean, NA(TDict(TInt32, TInt32)), 1), null)
  }
}

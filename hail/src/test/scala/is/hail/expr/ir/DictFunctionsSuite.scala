package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual._
import is.hail.utils.FastSeq

import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}

class DictFunctionsSuite extends HailSuite {
  def tuplesToMap(a: IndexedSeq[(Integer, Integer)]): Map[Integer, Integer] =
    Option(a).map(_.filter(_ != null).toMap).orNull

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(IndexedSeq((1, 3), (2, 7))),
    Array(IndexedSeq((1, 3), (2, null), null, (null, 1), (3, 7))),
    Array(IndexedSeq()),
    Array(IndexedSeq(null)),
    Array(null),
  )

  @Test(dataProvider = "basic")
  def dictFromArray(a: IndexedSeq[(Integer, Integer)]): Unit = {
    assertEvalsTo(invoke("dict", TDict(TInt32, TInt32), toIRPairArray(a)), tuplesToMap(a))
    assertEvalsTo(toIRDict(a), tuplesToMap(a))
  }

  @Test(dataProvider = "basic")
  def dictFromSet(a: IndexedSeq[(Integer, Integer)]): Unit = {
    assertEvalsTo(
      invoke("dict", TDict(TInt32, TInt32), ToSet(ToStream(toIRPairArray(a)))),
      tuplesToMap(a),
    )
  }

  @Test(dataProvider = "basic")
  def isEmpty(a: IndexedSeq[(Integer, Integer)]): Unit = {
    assertEvalsTo(
      invoke("isEmpty", TBoolean, toIRDict(a)),
      Option(a).map(_.forall(_ == null)).orNull,
    )
  }

  @DataProvider(name = "dictToArray")
  def dictToArrayData(): Array[Array[Any]] = Array(
    Array(FastSeq(1 -> 3, 2 -> 7), FastSeq(Row(1, 3), Row(2, 7))),
    Array(
      FastSeq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      FastSeq(Row(1, 3), Row(2, null), Row(3, 7), Row(null, 1)),
    ),
    Array(FastSeq(), FastSeq()),
    Array(FastSeq(null), FastSeq()),
    Array(null, null),
  )

  @Test(dataProvider = "dictToArray")
  def dictToArray(a: IndexedSeq[(Integer, Integer)], expected: (IndexedSeq[Row])): Unit = {
    assertEvalsTo(invoke("dictToArray", TArray(TTuple(TInt32, TInt32)), toIRDict(a)), expected)
  }

  @DataProvider(name = "keysAndValues")
  def keysAndValuesData(): Array[Array[Any]] = Array(
    Array(FastSeq(1 -> 3, 2 -> 7), FastSeq(1, 2), FastSeq(3, 7)),
    Array(
      FastSeq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      FastSeq(1, 2, 3, null),
      FastSeq(3, null, 7, 1),
    ),
    Array(FastSeq(), FastSeq(), FastSeq()),
    Array(FastSeq(null), FastSeq(), FastSeq()),
    Array(null, null, null),
  )

  @Test(dataProvider = "keysAndValues")
  def keySet(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  ): Unit = {
    assertEvalsTo(invoke("keySet", TSet(TInt32), toIRDict(a)), Option(keys).map(_.toSet).orNull)
  }

  @Test(dataProvider = "keysAndValues")
  def keys(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  ): Unit = {
    assertEvalsTo(invoke("keys", TArray(TInt32), toIRDict(a)), keys)
  }

  @Test(dataProvider = "keysAndValues")
  def values(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  ): Unit = {
    assertEvalsTo(invoke("values", TArray(TInt32), toIRDict(a)), values)
  }

  val d = IRDict((1, 3), (3, 7), (5, null), (null, 5))
  val dwoutna = IRDict((1, 3), (3, 7), (5, null))
  val na = NA(TInt32)

  @Test def dictGet(): Unit = {
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

  @Test def dictContains(): Unit = {
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

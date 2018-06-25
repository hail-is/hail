package is.hail.expr.ir

import is.hail.utils._
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types.{TDict, TInt32}
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class DictFunctionsSuite extends TestNGSuite {
  def tuplesToMap(a: Seq[(Integer, Integer)]): Map[Integer, Integer] =
    Option(a).map(_.filter(_ != null).toMap).orNull

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(Seq((1, 3), (2, 7))),
    Array(Seq((1, 3), (2, null), null, (null, 1), (3, 7))),
    Array(Seq()),
    Array(Seq(null)),
    Array(null)
  )

  @Test(dataProvider = "basic")
  def dictFromArray(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("dict", toIRPairArray(a)), tuplesToMap(a))
    assertEvalsTo(toIRDict(a), tuplesToMap(a))
  }

  @Test(dataProvider = "basic")
  def dictFromSet(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("dict", ToSet(toIRPairArray(a))), tuplesToMap(a))
  }

  @Test(dataProvider = "basic")
  def size(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("size", toIRDict(a)),
      Option(a).map(_.count(_ != null)).orNull)
  }

  @Test(dataProvider = "basic")
  def isEmpty(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("isEmpty", toIRDict(a)),
      Option(a).map(_.forall(_ == null)).orNull)
  }

  @DataProvider(name = "dictToArray")
  def dictToArrayData(): Array[Array[Any]] = Array(
    Array(Seq(1 -> 3, 2 -> 7), IndexedSeq(Row(1, 3), Row(2, 7))),
    Array(Seq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      IndexedSeq(Row(1, 3), Row(2, null), Row(3, 7), Row(null, 1))),
    Array(Seq(), IndexedSeq()),
    Array(Seq(null), IndexedSeq()),
    Array(null, null))

  @Test(dataProvider = "dictToArray")
  def dictToArray(a: Seq[(Integer, Integer)], expected: (IndexedSeq[Row])) {
    assertEvalsTo(invoke("dictToArray", toIRDict(a)), expected)
  }

  @DataProvider(name = "keysAndValues")
  def keysAndValuesData(): Array[Array[Any]] = Array(
    Array(Seq(1 -> 3, 2 -> 7), IndexedSeq(1, 2), IndexedSeq(3, 7)),
    Array(Seq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      IndexedSeq(1, 2, 3, null), IndexedSeq(3, null, 7, 1)),
    Array(Seq(), IndexedSeq(), IndexedSeq()),
    Array(Seq(null), IndexedSeq(), IndexedSeq()),
    Array(null, null, null))

  @Test(dataProvider = "keysAndValues")
  def keySet(a: Seq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("keySet", toIRDict(a)),
      Option(keys).map(_.toSet).orNull)
  }

  @Test(dataProvider = "keysAndValues")
  def keys(a: Seq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("keys", toIRDict(a)), keys)
  }

  @Test(dataProvider = "keysAndValues")
  def values(a: Seq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("values", toIRDict(a)), values)
  }

  @Test def dictGet() {
    val d = IRDict((1, 3), (2, 7), (3, null))
    val na = NA(TInt32())
    assertEvalsTo(invoke("get", NA(TDict(TInt32(), TInt32())), 1, na), null)
    assertEvalsTo(invoke("get", d, 1, na), 3)
    assertEvalsTo(invoke("get", d, 2, 50), 7)
    assertEvalsTo(invoke("get", d, 3, 50), null)
    assertEvalsTo(invoke("get", d, 100, 50), 50)
    assertEvalsTo(invoke("get", d, 100, na), null)

    assertEvalsTo(invoke("[]", d, 1), 3)
    assertEvalsTo(invoke("[]", d, 3), null)
    assertFatal(invoke("[]", d, 100), "dictionary")

    assertEvalsTo(invoke("get", d, na, 50), null)
    assertEvalsTo(invoke("[]", d, na), null)
  }
}

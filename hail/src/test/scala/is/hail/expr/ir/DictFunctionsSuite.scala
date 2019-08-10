package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.expr.types.virtual.{TDict, TInt32}
import is.hail.utils.FastIndexedSeq
import org.apache.spark.sql.Row
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class DictFunctionsSuite extends TestNGSuite {
  def tuplesToMap(a: Seq[(Integer, Integer)]): Map[Integer, Integer] =
    Option(a).map(_.filter(_ != null).toMap).orNull

  implicit val execStrats = ExecStrategy.javaOnly

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
  def isEmpty(a: Seq[(Integer, Integer)]) {
    assertEvalsTo(invoke("isEmpty", toIRDict(a)),
      Option(a).map(_.forall(_ == null)).orNull)
  }

  @DataProvider(name = "dictToArray")
  def dictToArrayData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(1 -> 3, 2 -> 7), FastIndexedSeq(Row(1, 3), Row(2, 7))),
    Array(FastIndexedSeq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      FastIndexedSeq(Row(1, 3), Row(2, null), Row(3, 7), Row(null, 1))),
    Array(FastIndexedSeq(), FastIndexedSeq()),
    Array(FastIndexedSeq(null), FastIndexedSeq()),
    Array(null, null))

  @Test(dataProvider = "dictToArray")
  def dictToArray(a: Seq[(Integer, Integer)], expected: (IndexedSeq[Row])) {
    assertEvalsTo(invoke("dictToArray", toIRDict(a)), expected)
  }

  @DataProvider(name = "keysAndValues")
  def keysAndValuesData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(1 -> 3, 2 -> 7), FastIndexedSeq(1, 2), FastIndexedSeq(3, 7)),
    Array(FastIndexedSeq(1 -> 3, 2 -> null, null, (null, 1), 3 -> 7),
      FastIndexedSeq(1, 2, 3, null), FastIndexedSeq(3, null, 7, 1)),
    Array(FastIndexedSeq(), FastIndexedSeq(), FastIndexedSeq()),
    Array(FastIndexedSeq(null), FastIndexedSeq(), FastIndexedSeq()),
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

  val d = IRDict((1, 3), (3, 7), (5, null), (null, 5))
  val dwoutna = IRDict((1, 3), (3, 7), (5, null))
  val na = NA(TInt32())

  @Test def dictGet() {
    assertEvalsTo(invoke("get", NA(TDict(TInt32(), TInt32())), 1, na), null)
    assertEvalsTo(invoke("get", d, 0, na), null)
    assertEvalsTo(invoke("get", d, 1, na), 3)
    assertEvalsTo(invoke("get", d, 2, na), null)
    assertEvalsTo(invoke("get", d, 3, 50), 7)
    assertEvalsTo(invoke("get", d, 4, -7), -7)
    assertEvalsTo(invoke("get", d, 5, 50), null)
    assertEvalsTo(invoke("get", d, na, 50), 5)
    assertEvalsTo(invoke("get", dwoutna, na, 50), 50)
    assertEvalsTo(invoke("get", d, 100, 50), 50)
    assertEvalsTo(invoke("get", d, 100, na), null)
    assertEvalsTo(invoke("get", dwoutna, 100, 50), 50)

    assertEvalsTo(invoke("get", IRDict(), 100, na), null)
    assertEvalsTo(invoke("get", IRDict(), 100, 50), 50)

    assertEvalsTo(invoke("[]", d, 1), 3)
    assertEvalsTo(invoke("[]", d, 5), null)
    assertEvalsTo(invoke("[]", d, na), 5)

    assertFatal(invoke("[]", d, -5), "dictionary")
    assertFatal(invoke("[]", d, 100), "dictionary")
    assertFatal(invoke("[]", IRDict(), 100), "dictionary")
  }

  @Test def dictContains() {
    assertEvalsTo(invoke("contains", d, 0), false)
    assertEvalsTo(invoke("contains", d, 1), true)
    assertEvalsTo(invoke("contains", d, 2), false)
    assertEvalsTo(invoke("contains", d, 3), true)
    assertEvalsTo(invoke("contains", d, 4), false)
    assertEvalsTo(invoke("contains", d, 5), true)
    assertEvalsTo(invoke("contains", d, 100), false)
    assertEvalsTo(invoke("contains", dwoutna, 100), false)
    assertEvalsTo(invoke("contains", d, na), true)

    assert(eval(invoke("contains", IRDict(), 100)) == false)
    assertEvalsTo(invoke("contains", NA(TDict(TInt32(), TInt32())), 1), null)
  }
}

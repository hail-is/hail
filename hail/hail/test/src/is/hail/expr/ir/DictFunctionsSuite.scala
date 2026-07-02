package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{NA, ToSet, ToStream}
import is.hail.types.virtual._

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class DictFunctionsSuite {
  def tuplesToMap(a: IndexedSeq[(Integer, Integer)]): Map[Integer, Integer] =
    Option(a).map(_.filter(_ != null).toMap).orNull

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  def basic() = ArraySeq(
    IndexedSeq((1, 3), (2, 7)),
    IndexedSeq((1, 3), (2, null), null, (null, 1), (3, 7)),
    IndexedSeq(),
    IndexedSeq(null),
    null,
  )

  @ParameterizedTest("basic")
  def dictFromArray(a: IndexedSeq[(Integer, Integer)])(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("dict", TDict(TInt32, TInt32), toIRPairArray(a)), tuplesToMap(a))
    assertEvalsTo(toIRDict(a), tuplesToMap(a))
  }

  @ParameterizedTest("basic")
  def dictFromSet(a: IndexedSeq[(Integer, Integer)])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("dict", TDict(TInt32, TInt32), ToSet(ToStream(toIRPairArray(a)))),
      tuplesToMap(a),
    )

  @ParameterizedTest("basic")
  def isEmpty(a: IndexedSeq[(Integer, Integer)])(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(
      invoke("isEmpty", TBoolean, toIRDict(a)),
      Option(a).map(_.forall(_ == null)).orNull,
    )

  def dictToArray() = ArraySeq[(IndexedSeq[(Integer, Integer)], IndexedSeq[Row])](
    (FastSeq[(Integer, Integer)]((1, 3), (2, 7)), FastSeq(RowSeq(1, 3), RowSeq(2, 7))),
    (
      FastSeq[(Integer, Integer)]((1, 3), (2, null), null, (null, 1), (3, 7)),
      FastSeq(RowSeq(1, 3), RowSeq(2, null), RowSeq(3, 7), RowSeq(null, 1)),
    ),
    (FastSeq(), FastSeq()),
    (FastSeq(null), FastSeq()),
    (null, null),
  )

  @ParameterizedTest
  def dictToArray(
    a: IndexedSeq[(Integer, Integer)],
    expected: IndexedSeq[Row],
  )(implicit
    ctx: ExecuteContext
  ): Unit = {
    implicit val execStrats = Set(ExecStrategy.JvmCompile)
    assertEvalsTo(invoke("dictToArray", TArray(TTuple(TInt32, TInt32)), toIRDict(a)), expected)
  }

  def keysAndValues() = ArraySeq[
    (IndexedSeq[(Integer, Integer)], IndexedSeq[Integer], IndexedSeq[Integer])
  ](
    (FastSeq[(Integer, Integer)]((1, 3), (2, 7)), FastSeq[Integer](1, 2), FastSeq[Integer](3, 7)),
    (
      FastSeq[(Integer, Integer)]((1, 3), (2, null), null, (null, 1), (3, 7)),
      FastSeq[Integer](1, 2, 3, null),
      FastSeq[Integer](3, null, 7, 1),
    ),
    (FastSeq(), FastSeq(), FastSeq()),
    (FastSeq(null), FastSeq(), FastSeq()),
    (null, null, null),
  )

  @ParameterizedTest("keysAndValues")
  def keySet(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("keySet", TSet(TInt32), toIRDict(a)), Option(keys).map(_.toSet).orNull)

  @ParameterizedTest("keysAndValues")
  def keys(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("keys", TArray(TInt32), toIRDict(a)), keys)

  @ParameterizedTest("keysAndValues")
  def values(
    a: IndexedSeq[(Integer, Integer)],
    keys: IndexedSeq[Integer],
    values: IndexedSeq[Integer],
  )(implicit ctx: ExecuteContext
  ): Unit =
    assertEvalsTo(invoke("values", TArray(TInt32), toIRDict(a)), values)

  val d = IRDict((1, 3), (3, 7), (5, null), (null, 5))
  val dwoutna = IRDict((1, 3), (3, 7), (5, null))
  val na = NA(TInt32)

  @Test def dictGet(implicit ctx: ExecuteContext): Unit = {
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

  @Test def dictContains(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("contains", TBoolean, d, 0), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 1), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 2), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 3), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 4), false)
    assertEvalsTo(invoke("contains", TBoolean, d, 5), true)
    assertEvalsTo(invoke("contains", TBoolean, d, 100), false)
    assertEvalsTo(invoke("contains", TBoolean, dwoutna, 100), false)
    assertEvalsTo(invoke("contains", TBoolean, d, na), true)

    assertEq(eval(invoke("contains", TBoolean, IRDict(), 100)), false)
    assertEvalsTo(invoke("contains", TBoolean, NA(TDict(TInt32, TInt32)), 1), null)
  }
}

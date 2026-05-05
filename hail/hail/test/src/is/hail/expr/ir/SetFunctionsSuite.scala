package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{I32, NA, ToSet, ToStream}
import is.hail.types.virtual._

import org.testng.annotations.{DataProvider, Test}

class SetFunctionsSuite extends HailSuite {
  val naa = NA(TArray(TInt32))
  val nas = NA(TSet(TInt32))

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  @Test def toSet(): Unit = {
    assertEvalsTo(IRSet(3, 7), Set(3, 7))
    assertEvalsTo(IRSet(3, null, 7), Set(null, 3, 7))
    assertEvalsTo(nas, null)
    assertEvalsTo(ToSet(ToStream(naa)), null)
    assertEvalsTo(invoke("toSet", TSet(TInt32), IRArray(3, 7)), Set(3, 7))
    assertEvalsTo(invoke("toSet", TSet(TInt32), IRArray(3, null, 7)), Set(null, 3, 7))
    assertEvalsTo(invoke("toSet", TSet(TInt32), naa), null)
  }

  @Test def isEmpty(): Unit = {
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(3, 7)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(3, null, 7)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet()), true)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(null)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, nas), null)
  }

  @Test def contains(): Unit = {
    val s = IRSet(3, null, 7)
    val swoutna = IRSet(3, 7)

    assertEvalsTo(invoke("contains", TBoolean, s, I32(3)), true)
    assertEvalsTo(invoke("contains", TBoolean, s, I32(4)), false)
    assertEvalsTo(invoke("contains", TBoolean, s, I32(10)), false)
    assertEvalsTo(invoke("contains", TBoolean, swoutna, I32(10)), false)
    assertEvalsTo(invoke("contains", TBoolean, s, NA(TInt32)), true)
    assertEvalsTo(invoke("contains", TBoolean, swoutna, NA(TInt32)), false)
    assertEvalsTo(invoke("contains", TBoolean, IRSet(3, 7), NA(TInt32)), false)
    assert(eval(invoke("contains", TBoolean, IRSet(), 3)) == false)
  }

  @Test def remove(): Unit = {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("remove", TSet(TInt32), s, I32(3)), Set(null, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), s, I32(4)), Set(null, 3, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), s, NA(TInt32)), Set(3, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), IRSet(3, 7), NA(TInt32)), Set(3, 7))
  }

  @Test def add(): Unit = {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(3)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, NA(TInt32)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), IRSet(3, 7), NA(TInt32)), Set(null, 3, 7))
  }

  @DataProvider(name = "IsSubset")
  def dataIsSubset: Array[Array[Any]] =
    Array(
      Array(IRSet(), IRSet(), true),
      Array(IRSet(1), IRSet(1), true),
      Array(IRSet(3, null, 7), IRSet(3, null, 7), true),
      Array(IRSet(3, null, 7), IRSet(3, null, 7, 11), true),
      Array(IRSet(1, 2, 3), IRSet(1, 2, 4), false),
      Array(IRSet(1, 2, 3), NA(TSet(TInt32)), null),
      Array(NA(TSet(TInt32)), IRSet(1, 2, 3), null),
      Array(NA(TSet(TInt32)), NA(TSet(TInt32)), null),
    )

  @Test(dataProvider = "IsSubset")
  def testIsSubset(a: IR, b: IR, isSubset: Any): Unit =
    assertEvalsTo(invoke("isSubset", TBoolean, a, b), isSubset)

  @Test def union(): Unit = {
    assertEvalsTo(invoke("union", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(null, 3, 7, 8))
    assertEvalsTo(invoke("union", TSet(TInt32), IRSet(3, 7), IRSet(3, 8, null)), Set(null, 3, 7, 8))
  }

  @Test def intersection(): Unit = {
    assertEvalsTo(invoke("intersection", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(3))
    assertEvalsTo(
      invoke("intersection", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8, null)),
      Set(null, 3),
    )
  }

  @Test def difference(): Unit = {
    assertEvalsTo(invoke("difference", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(null, 7))
    assertEvalsTo(invoke("difference", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8, null)), Set(7))
  }

  @Test def median(): Unit = {
    assertEvalsTo(invoke("median", TInt32, IRSet(5)), 5)
    assertEvalsTo(invoke("median", TInt32, IRSet(5, null)), 5)
    assertEvalsTo(invoke("median", TInt32, IRSet(3, 7)), 5)
    assertEvalsTo(invoke("median", TInt32, IRSet(3, null, 7, 1)), 3)
    assertEvalsTo(invoke("median", TInt32, IRSet(3, 7, 1)), 3)
    assertEvalsTo(invoke("median", TInt32, IRSet(3, null, 9, 6, 1)), 4)
    assertEvalsTo(invoke("median", TInt32, IRSet()), null)
    assertEvalsTo(invoke("median", TInt32, IRSet(null)), null)
    assertEvalsTo(invoke("median", TInt32, nas), null)
  }
}

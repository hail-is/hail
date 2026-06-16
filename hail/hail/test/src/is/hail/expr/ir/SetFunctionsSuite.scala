package is.hail.expr.ir

import is.hail.{ExecStrategy, ParameterizedTest}
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.TestUtils._
import is.hail.expr.ir.defs.{I32, NA, ToSet, ToStream}
import is.hail.types.virtual._

import org.junit.jupiter.api.Test

class SetFunctionsSuite {
  val naa = NA(TArray(TInt32))
  val nas = NA(TSet(TInt32))

  implicit val execStrats: Set[ExecStrategy] = ExecStrategy.javaOnly

  @Test def toSet(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(IRSet(3, 7), Set(3, 7))
    assertEvalsTo(IRSet(3, null, 7), Set(null, 3, 7))
    assertEvalsTo(nas, null)
    assertEvalsTo(ToSet(ToStream(naa)), null)
    assertEvalsTo(invoke("toSet", TSet(TInt32), IRArray(3, 7)), Set(3, 7))
    assertEvalsTo(invoke("toSet", TSet(TInt32), IRArray(3, null, 7)), Set(null, 3, 7))
    assertEvalsTo(invoke("toSet", TSet(TInt32), naa), null)
  }

  @Test def isEmpty(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(3, 7)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(3, null, 7)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet()), true)
    assertEvalsTo(invoke("isEmpty", TBoolean, IRSet(null)), false)
    assertEvalsTo(invoke("isEmpty", TBoolean, nas), null)
  }

  @Test def contains(implicit ctx: ExecuteContext): Unit = {
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

  @Test def remove(implicit ctx: ExecuteContext): Unit = {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("remove", TSet(TInt32), s, I32(3)), Set(null, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), s, I32(4)), Set(null, 3, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), s, NA(TInt32)), Set(3, 7))
    assertEvalsTo(invoke("remove", TSet(TInt32), IRSet(3, 7), NA(TInt32)), Set(3, 7))
  }

  @Test def add(implicit ctx: ExecuteContext): Unit = {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(3)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), s, NA(TInt32)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", TSet(TInt32), IRSet(3, 7), NA(TInt32)), Set(null, 3, 7))
  }

  def testIsSubset = ArraySeq[(IR, IR, Any)](
    (IRSet(), IRSet(), true),
    (IRSet(1), IRSet(1), true),
    (IRSet(3, null, 7), IRSet(3, null, 7), true),
    (IRSet(3, null, 7), IRSet(3, null, 7, 11), true),
    (IRSet(1, 2, 3), IRSet(1, 2, 4), false),
    (IRSet(1, 2, 3), NA(TSet(TInt32)), null),
    (NA(TSet(TInt32)), IRSet(1, 2, 3), null),
    (NA(TSet(TInt32)), NA(TSet(TInt32)), null),
  )

  @ParameterizedTest
  def testIsSubset(a: IR, b: IR, isSubset: Any)(implicit ctx: ExecuteContext): Unit =
    assertEvalsTo(invoke("isSubset", TBoolean, a, b), isSubset)

  @Test def union(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("union", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(null, 3, 7, 8))
    assertEvalsTo(invoke("union", TSet(TInt32), IRSet(3, 7), IRSet(3, 8, null)), Set(null, 3, 7, 8))
  }

  @Test def intersection(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("intersection", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(3))
    assertEvalsTo(
      invoke("intersection", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8, null)),
      Set(null, 3),
    )
  }

  @Test def difference(implicit ctx: ExecuteContext): Unit = {
    assertEvalsTo(invoke("difference", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8)), Set(null, 7))
    assertEvalsTo(invoke("difference", TSet(TInt32), IRSet(3, null, 7), IRSet(3, 8, null)), Set(7))
  }

  @Test def median(implicit ctx: ExecuteContext): Unit = {
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

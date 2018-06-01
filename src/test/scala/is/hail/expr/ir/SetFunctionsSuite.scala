package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._

import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class SetFunctionsSuite extends TestNGSuite {
  val naa = NA(TArray(TInt32()))
  val nas = NA(TSet(TInt32()))

  @Test def toSet() {
    assertEvalsTo(IRSet(3, 7), Set(3, 7))
    assertEvalsTo(IRSet(3, null, 7), Set(null, 3, 7))
    assertEvalsTo(nas, null)
    assertEvalsTo(ToSet(naa), null)
    assertEvalsTo(invoke("toSet", IRArray(3, 7)), Set(3, 7))
    assertEvalsTo(invoke("toSet", IRArray(3, null, 7)), Set(null, 3, 7))
    assertEvalsTo(invoke("toSet", naa), null)
  }

  @Test def size() {
    assertEvalsTo(invoke("size", IRSet(3, 7)), 2)
    assertEvalsTo(invoke("size", IRSet(3, null, 7)), 3)
    assertEvalsTo(invoke("size", IRSet()), 0)
    assertEvalsTo(invoke("size", IRSet(null)), 1)
    assertEvalsTo(invoke("size", nas), null)
  }

  @Test def isEmpty() {
    assertEvalsTo(invoke("isEmpty", IRSet(3, 7)), false)
    assertEvalsTo(invoke("isEmpty", IRSet(3, null, 7)), false)
    assertEvalsTo(invoke("isEmpty", IRSet()), true)
    assertEvalsTo(invoke("isEmpty", IRSet(null)), false)
    assertEvalsTo(invoke("isEmpty", nas), null)
  }

  @Test def contains() {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("contains", s, I32(3)), true)
    assertEvalsTo(invoke("contains", s, I32(4)), false)
    assertEvalsTo(invoke("contains", s, NA(TInt32())), true)
    assertEvalsTo(invoke("contains", IRSet(3, 7), NA(TInt32())), false)
  }

  @Test def remove() {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("remove", s, I32(3)), Set(null, 7))
    assertEvalsTo(invoke("remove", s, I32(4)), Set(null, 3, 7))
    assertEvalsTo(invoke("remove", s, NA(TInt32())), Set(3, 7))
    assertEvalsTo(invoke("remove", IRSet(3, 7), NA(TInt32())), Set(3, 7))
  }

  @Test def add() {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("add", s, I32(3)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", s, NA(TInt32())), Set(null, 3, 7))
    assertEvalsTo(invoke("add", IRSet(3, 7), NA(TInt32())), Set(null, 3, 7))
  }

  @Test def isSubset() {
    val s = IRSet(3, null, 7)
    assertEvalsTo(invoke("isSubset", s, invoke("add", s, I32(4))), true)
    assertEvalsTo(invoke("isSubset", IRSet(3, 7), invoke("add", IRSet(3, 7), NA(TInt32()))), true)
    assertEvalsTo(invoke("isSubset", s, invoke("remove", s, I32(3))), false)
    assertEvalsTo(invoke("isSubset", s, invoke("remove", s, NA(TInt32()))), false)
  }

  @Test def union() {
    assertEvalsTo(invoke("union", IRSet(3, null, 7), IRSet(3, 8)), Set(null, 3, 7, 8))
    assertEvalsTo(invoke("union", IRSet(3, 7), IRSet(3, 8, null)), Set(null, 3, 7, 8))
  }

  @Test def intersection() {
    assertEvalsTo(invoke("intersection", IRSet(3, null, 7), IRSet(3, 8)), Set(3))
    assertEvalsTo(invoke("intersection", IRSet(3, null, 7), IRSet(3, 8, null)), Set(null, 3))
  }

  @Test def difference() {
    assertEvalsTo(invoke("difference", IRSet(3, null, 7), IRSet(3, 8)), Set(null, 7))
    assertEvalsTo(invoke("difference", IRSet(3, null, 7), IRSet(3, 8, null)), Set(7))
  }

  @Test def sum() {
    assertEvalsTo(invoke("sum", IRSet(3, 7)), 10)
    assertEvalsTo(invoke("sum", IRSet(3, null, 7)), 10)
    assertEvalsTo(invoke("sum", IRSet()), 0)
    assertEvalsTo(invoke("sum", IRSet(null)), 0)
    assertEvalsTo(invoke("sum", nas), null)
  }

  @Test def product() {
    assertEvalsTo(invoke("product", IRSet(3, 7)), 21)
    assertEvalsTo(invoke("product", IRSet(3, null, 7)), 21)
    assertEvalsTo(invoke("product", IRSet()), 1)
    assertEvalsTo(invoke("product", IRSet(null)), 1)
    assertEvalsTo(invoke("product", nas), null)
  }
}

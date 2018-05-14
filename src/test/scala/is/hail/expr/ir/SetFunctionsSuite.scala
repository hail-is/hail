package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._

import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class SetFunctionsSuite extends TestNGSuite {
  val naa = NA(TArray(TInt32()))
  val a0 = MakeArray(Seq(I32(3), I32(7)), TArray(TInt32()))
  val s0 = ToSet(a0)
  val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7)), TArray(TInt32()))
  val s = ToSet(a)
  val nas = NA(TSet(TInt32()))
  val t = ToSet(MakeArray(Seq(I32(3), I32(8)), TArray(TInt32())))
  val t2 = ToSet(MakeArray(Seq(I32(3), I32(8), NA(TInt32())), TArray(TInt32())))

  @Test def toSet() {
    assertEvalsTo(s0, Set(3, 7))
    assertEvalsTo(s, Set(null, 3, 7))
    assertEvalsTo(nas, null)
    assertEvalsTo(ToSet(naa), null)
    assertEvalsTo(invoke("toSet", a0), Set(3, 7))
    assertEvalsTo(invoke("toSet", a), Set(null, 3, 7))
    assertEvalsTo(invoke("toSet", naa), null)
  }

  @Test def contains() {
    assertEvalsTo(invoke("contains", s, I32(3)), true)
    assertEvalsTo(invoke("contains", s, I32(4)), false)
    assertEvalsTo(invoke("contains", s, NA(TInt32())), true)
    assertEvalsTo(invoke("contains", s0, NA(TInt32())), false)
  }

  @Test def remove() {
    assertEvalsTo(invoke("remove", s, I32(3)), Set(null, 7))
    assertEvalsTo(invoke("remove", s, I32(4)), Set(null, 3, 7))
    assertEvalsTo(invoke("remove", s, NA(TInt32())), Set(3, 7))
    assertEvalsTo(invoke("remove", s0, NA(TInt32())), Set(3, 7))
  }

  @Test def add() {
    assertEvalsTo(invoke("add", s, I32(3)), Set(null, 3, 7))
    assertEvalsTo(invoke("add", s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", s, I32(4)), Set(null, 3, 4, 7))
    assertEvalsTo(invoke("add", s, NA(TInt32())), Set(null, 3, 7))
    assertEvalsTo(invoke("add", s0, NA(TInt32())), Set(null, 3, 7))
  }

  @Test def isSubset() {
    assertEvalsTo(invoke("isSubset", s, invoke("add", s, I32(4))), true)
    assertEvalsTo(invoke("isSubset", s0, invoke("add", s0, NA(TInt32()))), true)
    assertEvalsTo(invoke("isSubset", s, invoke("remove", s, I32(3))), false)
    assertEvalsTo(invoke("isSubset", s, invoke("remove", s, NA(TInt32()))), false)
  }

  @Test def union() {
    assertEvalsTo(invoke("union", s, t), Set(null, 3, 7, 8))
    assertEvalsTo(invoke("union", s0, t2), Set(null, 3, 7, 8))
  }

  @Test def intersection() {
    assertEvalsTo(invoke("intersection", s, t), Set(3))
    assertEvalsTo(invoke("intersection", s, t2), Set(null, 3))
  }

  @Test def difference() {
    assertEvalsTo(invoke("difference", s, t), Set(null, 7))
    assertEvalsTo(invoke("difference", s, t2), Set(7))
  }
}

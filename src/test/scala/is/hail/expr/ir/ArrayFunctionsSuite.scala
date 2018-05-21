package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._

import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class ArrayFunctionsSuite extends TestNGSuite {
  val naa = NA(TArray(TInt32()))
  val a0 = MakeArray(Seq(I32(3), I32(7)), TArray(TInt32()))
  val a = MakeArray(Seq(I32(3), NA(TInt32()), I32(7), NA(TInt32())), TArray(TInt32()))
  val b = MakeArray(Seq(I32(3), NA(TInt32()), I32(3), NA(TInt32()), I32(1)), TArray(TInt32()))
  val e = MakeArray(Seq(), TArray(TInt32()))

  @Test def length() {
    assertEvalsTo(invoke("length", a0), 2)
    assertEvalsTo(invoke("length", naa), null)
    assertEvalsTo(invoke("length", a), 4)
    assertEvalsTo(invoke("length", e), 0)
  }

  @Test def size() {
    assertEvalsTo(invoke("size", a0), 2)
    assertEvalsTo(invoke("size", naa), null)
    assertEvalsTo(invoke("size", a), 4)
    assertEvalsTo(invoke("size", e), 0)
  }

  @Test def isEmpty() {
    assertEvalsTo(invoke("isEmpty", a0), false)
    assertEvalsTo(invoke("isEmpty", naa), null)
    assertEvalsTo(invoke("isEmpty", a), false)
    assertEvalsTo(invoke("isEmpty", e), true)
  }

  @Test def sort() {
    assertEvalsTo(invoke("sort", a0), Seq(3, 7))
    assertEvalsTo(invoke("sort", a0, True()), Seq(3, 7))
    assertEvalsTo(invoke("sort", a0, False()), Seq(7, 3))
    assertEvalsTo(invoke("sort", naa), null)
    assertEvalsTo(invoke("sort", a), Seq(3, 7, null, null))
    assertEvalsTo(invoke("sort", a, False()), Seq(7, 3, null, null))
    assertEvalsTo(invoke("sort", b), Seq(1, 3, 3, null, null))
    assertEvalsTo(invoke("sort", b, False()), Seq(3, 3, 1, null, null))
    assertEvalsTo(invoke("sort", e), Seq())
  }

  @Test def append() {
    assertEvalsTo(invoke("append", a0, I32(1)), Seq(3, 7, 1))
    assertEvalsTo(invoke("append", a0, NA(TInt32())), Seq(3, 7, null))
    assertEvalsTo(invoke("append", naa, I32(1)), null)
    assertEvalsTo(invoke("append", a, I32(1)), Seq(3, null, 7, null, 1))
    assertEvalsTo(invoke("append", e, I32(1)), Seq(1))
  }

  @Test def extend() {
    assertEvalsTo(invoke("extend", a0, a0), Seq(3, 7, 3, 7))
    assertEvalsTo(invoke("extend", a0, naa), null)
    assertEvalsTo(invoke("extend", naa, a0), null)
    assertEvalsTo(invoke("extend", a, a0), Seq(3, null, 7, null, 3, 7))
    assertEvalsTo(invoke("extend", a, e), Seq(3, null, 7, null))
    assertEvalsTo(invoke("extend", e, a), Seq(3, null, 7, null))
    assertEvalsTo(invoke("extend", e, e), Seq())
  }

  @Test def sum() {
    assertEvalsTo(invoke("sum", a0), 10)
    assertEvalsTo(invoke("sum", naa), null)
    assertEvalsTo(invoke("sum", a), 10)
    assertEvalsTo(invoke("sum", e), 0)
  }

  @Test def product() {
    assertEvalsTo(invoke("product", a0), 21)
    assertEvalsTo(invoke("product", naa), null)
    assertEvalsTo(invoke("product", a), 21)
    assertEvalsTo(invoke("product", e), 1)
  }

  @Test def arrayOps() {
    assertEvalsTo(invoke("+", a0, a0), Seq(6, 14))
    assertEvalsTo(invoke("+", a0, a), Seq(6, null))
    assertEvalsTo(invoke("+", naa, a), null)
    assertEvalsTo(invoke("+", a, naa), null)
    assertEvalsTo(invoke("+", I32(1), a), Seq(4, null, 8, null))
  }
}

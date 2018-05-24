package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class ArrayFunctionsSuite extends TestNGSuite {
  def IRArray(a: Seq[Integer]): IR =
    if (a == null)
      NA(TArray(TInt32()))
    else
      MakeArray(a.map { ji => Option(ji).map(I32(_)).getOrElse(NA(TInt32())) }, TArray(TInt32()))

  val naa = NA(TArray(TInt32()))
  val a0 = IRArray(Seq(3, 7))
  val a = IRArray(Seq(3, null, 7, null))
  val b = IRArray(Seq(3, null, 3, null, 1))
  val c = IRArray(Seq(2, null, null, 5, 7))
  val d = IRArray(Seq(1, null, 1, null, 3))
  val e = IRArray(Seq())

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
    assertEvalsTo(invoke("sort", a0), IndexedSeq(3, 7))
    assertEvalsTo(invoke("sort", a0, True()), IndexedSeq(3, 7))
    assertEvalsTo(invoke("sort", a0, False()), IndexedSeq(7, 3))
    assertEvalsTo(invoke("sort", naa), null)
    assertEvalsTo(invoke("sort", a), IndexedSeq(3, 7, null, null))
    assertEvalsTo(invoke("sort", a, False()), IndexedSeq(7, 3, null, null))
    assertEvalsTo(invoke("sort", b), IndexedSeq(1, 3, 3, null, null))
    assertEvalsTo(invoke("sort", b, False()), IndexedSeq(3, 3, 1, null, null))
    assertEvalsTo(invoke("sort", e), IndexedSeq())
  }

  @Test def append() {
    assertEvalsTo(invoke("append", a0, I32(1)), IndexedSeq(3, 7, 1))
    assertEvalsTo(invoke("append", a0, NA(TInt32())), IndexedSeq(3, 7, null))
    assertEvalsTo(invoke("append", naa, I32(1)), null)
    assertEvalsTo(invoke("append", a, I32(1)), IndexedSeq(3, null, 7, null, 1))
    assertEvalsTo(invoke("append", e, I32(1)), IndexedSeq(1))
  }

  @Test def extend() {
    assertEvalsTo(invoke("extend", a0, a0), IndexedSeq(3, 7, 3, 7))
    assertEvalsTo(invoke("extend", a0, naa), null)
    assertEvalsTo(invoke("extend", naa, a0), null)
    assertEvalsTo(invoke("extend", a, a0), IndexedSeq(3, null, 7, null, 3, 7))
    assertEvalsTo(invoke("extend", a, e), IndexedSeq(3, null, 7, null))
    assertEvalsTo(invoke("extend", e, a), IndexedSeq(3, null, 7, null))
    assertEvalsTo(invoke("extend", e, e), IndexedSeq())
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

  @Test def min() {
    assertEvalsTo(invoke("min", a0), 3)
    assertEvalsTo(invoke("min", naa), null)
    assertEvalsTo(invoke("min", a), 3)
    assertEvalsTo(invoke("min", b), 1)
    assertEvalsTo(invoke("min", e), null)
  }

  @Test def argmin() {
    assertEvalsTo(invoke("argmin", a0), 0)
    assertEvalsTo(invoke("argmin", naa), null)
    assertEvalsTo(invoke("argmin", a), 0)
    assertEvalsTo(invoke("argmin", b), 4)
    assertEvalsTo(invoke("argmin", d), 0)
    assertEvalsTo(invoke("argmin", e), null)

    assertEvalsTo(invoke("uniqueMinIndex", a0), 0)
    assertEvalsTo(invoke("uniqueMinIndex", naa), null)
    assertEvalsTo(invoke("uniqueMinIndex", a), 0)
    assertEvalsTo(invoke("uniqueMinIndex", b), 4)
    assertEvalsTo(invoke("uniqueMinIndex", d), null)
    assertEvalsTo(invoke("uniqueMinIndex", e), null)
  }

  @Test def max() {
    assertEvalsTo(invoke("max", a0), 7)
    assertEvalsTo(invoke("max", naa), null)
    assertEvalsTo(invoke("max", a), 7)
    assertEvalsTo(invoke("max", b), 3)
    assertEvalsTo(invoke("max", e), null)
  }

  @Test def argmax() {
    assertEvalsTo(invoke("argmax", a0), 1)
    assertEvalsTo(invoke("argmax", naa), null)
    assertEvalsTo(invoke("argmax", a), 2)
    assertEvalsTo(invoke("argmax", b), 0)
    assertEvalsTo(invoke("argmax", d), 4)
    assertEvalsTo(invoke("argmax", e), null)

    assertEvalsTo(invoke("uniqueMaxIndex", a0), 1)
    assertEvalsTo(invoke("uniqueMaxIndex", naa), null)
    assertEvalsTo(invoke("uniqueMaxIndex", a), 2)
    assertEvalsTo(invoke("uniqueMaxIndex", b), null)
    assertEvalsTo(invoke("uniqueMaxIndex", d), 4)
    assertEvalsTo(invoke("uniqueMaxIndex", e), null)
  }

  @Test def arrayOps() {
    assertEvalsTo(invoke("+", b, c), IndexedSeq(5, null, null, null, 8))
    assertFatal(invoke("+", a0, a), "Arrays must have same length")
    assertFatal(invoke("+", a, a0), "Arrays must have same length")
    assertEvalsTo(invoke("+", naa, a), null)
    assertEvalsTo(invoke("+", a, naa), null)
    assertEvalsTo(invoke("+", I32(1), a), IndexedSeq(4, null, 8, null))
    assertEvalsTo(invoke("+", a, I32(1)), IndexedSeq(4, null, 8, null))
    assertEvalsTo(invoke("+", NA(TInt32()), a), IndexedSeq(null, null, null, null))
    assertEvalsTo(invoke("+", a, NA(TInt32())), IndexedSeq(null, null, null, null))

    assertEvalsTo(invoke("-", b, c), IndexedSeq(1, null, null, null, -6))
    assertEvalsTo(invoke("*", b, c), IndexedSeq(6, null, null, null, 7))
    assertEvalsTo(invoke("/", b, c), IndexedSeq(3f/2, null, null, null, 1f/7))
    assertEvalsTo(invoke("//", b, c), IndexedSeq(1, null, null, null, 0))
    assertEvalsTo(invoke("**", b, c), IndexedSeq[java.lang.Double](9, null, null, null, 1))
    assertEvalsTo(invoke("%", b, c), IndexedSeq(1, null, null, null, 1))
  }

  @Test def slicing() {
    assertEvalsTo(invoke("[]", a, I32(0)), 3)
    assertEvalsTo(invoke("[]", a, I32(1)), null)
    assertEvalsTo(invoke("[]", a, I32(2)), 7)
    assertFatal(invoke("[]", a, I32(4)), "array index out of bounds")
    assertFatal(invoke("[]", a, I32(-1)), "array index out of bounds")
    assertEvalsTo(invoke("[]", naa, I32(2)), null)

    assertEvalsTo(invoke("[:]", a), IndexedSeq(3, null, 7, null))
    assertEvalsTo(invoke("[:]", naa), null)

    assertEvalsTo(invoke("[*:]", a, I32(1)), IndexedSeq(null, 7, null))
    assertEvalsTo(invoke("[*:]", a, I32(-2)), IndexedSeq(7, null))
    assertEvalsTo(invoke("[*:]", naa, I32(1)), null)
    assertEvalsTo(invoke("[*:]", a, I32(5)), IndexedSeq())
    assertEvalsTo(invoke("[*:]", a, I32(-5)), IndexedSeq(3, null, 7, null))
  }
}

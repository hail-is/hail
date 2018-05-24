package is.hail.expr.ir

import is.hail.expr.types._
import is.hail.TestUtils._
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class ArrayFunctionsSuite extends TestNGSuite {
  def toIRArray(a: Seq[Integer]): IR =
    if (a == null)
      NA(TArray(TInt32()))
    else
      MakeArray(a.map { ji => Option(ji).map(I32(_)).getOrElse(NA(TInt32())) }, TArray(TInt32()))

  def IRArray(a: Integer*): IR = toIRArray(a)

  val naa = NA(TArray(TInt32()))

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(3, 7)),
    Array(null),
    Array(IndexedSeq(3, null, 7, null)),
    Array(IndexedSeq())
  )

  @DataProvider(name = "basicPairs")
  def basicPairsData(): Array[Array[Any]] = basicData().flatten.combinations(2).toArray

  @Test(dataProvider = "basic")
  def length(a: Seq[Integer]) {
    assertEvalsTo(invoke("length", toIRArray(a)),
      Option(a).map(_.length).orNull)
  }

  @Test(dataProvider = "basic")
  def size(a: Seq[Integer]) {
    assertEvalsTo(invoke("size", toIRArray(a)),
      Option(a).map(_.length).orNull)
  }

  @Test(dataProvider = "basic")
  def isEmpty(a: Seq[Integer]) {
    assertEvalsTo(invoke("isEmpty", toIRArray(a)),
      Option(a).map(_.isEmpty).orNull)
  }

  @Test(dataProvider = "basic")
  def append(a: Seq[Integer]) {
    assertEvalsTo(invoke("append", toIRArray(a), I32(1)),
      Option(a).map(_ :+ 1).orNull)
  }

  @Test(dataProvider = "basic")
  def appendNull(a: Seq[Integer]) {
    assertEvalsTo(invoke("append", toIRArray(a), NA(TInt32())),
      Option(a).map(_ :+ null).orNull)
  }

  @Test(dataProvider = "basic")
  def sum(a: Seq[Integer]) {
    assertEvalsTo(invoke("sum", toIRArray(a)),
      Option(a).map(_.filter(_ != null).map(_.toInt).sum).orNull)
  }

  @Test(dataProvider = "basic")
  def product(a: Seq[Integer]) {
    assertEvalsTo(invoke("product", toIRArray(a)),
      Option(a).map(_.filter(_ != null).map(_.toInt).product).orNull)
  }

  @Test(dataProvider = "basicPairs")
  def extend(a: Seq[Integer], b: Seq[Integer]) {
    assertEvalsTo(invoke("extend", toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (x, y) => x ++ y}.orNull)
  }

  @DataProvider(name = "sort")
  def sortData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(3, 9, 7), IndexedSeq(3, 7, 9), IndexedSeq(9, 7, 3)),
    Array(null, null, null),
    Array(IndexedSeq(3, null, 1, null, 3), IndexedSeq(1, 3, 3, null, null), IndexedSeq(3, 3, 1, null, null)),
    Array(IndexedSeq(1, null, 3, null, 1), IndexedSeq(1, 1, 3, null, null), IndexedSeq(3, 1, 1, null, null)),
    Array(IndexedSeq(), IndexedSeq(), IndexedSeq())
  )

  @Test(dataProvider = "sort")
  def sort(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("sort", toIRArray(a)), asc)
  }

  @Test(dataProvider = "sort")
  def sortAsc(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("sort", toIRArray(a), True()), asc)
  }

  @Test(dataProvider = "sort")
  def sortDesc(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("sort", toIRArray(a), False()), desc)
  }

  @Test(dataProvider = "sort")
  def min(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("min", toIRArray(a)),
      Option(asc).flatMap(_.headOption).orNull)
  }

  @DataProvider(name = "argminmax")
  def argMinMaxData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(IndexedSeq(3, null, 1, null, 3), 2, 0),
    Array(IndexedSeq(1, null, 3, null, 1), 0, 2),
    Array(IndexedSeq(), null, null)
  )

  @Test(dataProvider = "argminmax")
  def argmin(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmin", toIRArray(a)), argmin)
  }

  @Test(dataProvider = "argminmax")
  def argmax(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmax", toIRArray(a)), argmax)
  }

  @DataProvider(name = "uniqueMinMaxIndex")
  def uniqueMinMaxData(): Array[Array[Any]] = Array(
    Array(IndexedSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(IndexedSeq(3, null, 1, null, 3), 2, null),
    Array(IndexedSeq(1, null, 3, null, 1), null, 2),
    Array(IndexedSeq(), null, null)
  )

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMinIndex(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMinIndex", toIRArray(a)), argmin)
  }

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMaxIndex(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMaxIndex", toIRArray(a)), argmax)
  }

  def arrayOpsData(): Array[Array[Any]] = Array[Any](
    IndexedSeq(3, 9, 7, 1),
    IndexedSeq(null, 2, null, 8),
    IndexedSeq(5, 3, null, null),
    null
  ).combinations(2).toArray

  @DataProvider(name = "arrayOpsOperations")
  def arrayOpsOperations: Array[Array[Any]] = Array[(String, (Int, Int) => Int)](
    ("+", _ + _),
    ("-", _ - _),
    ("*", _ * _),
    ("//", _ / _),
    ("**", math.pow(_, _).toInt),
    ("%", _ % _)
  ).map(_.productIterator.toArray)

  @DataProvider(name = "arrayOps")
  def arrayOpsPairs(): Array[Array[Any]] =
    for (Array(a, b) <- arrayOpsData(); Array(s, f) <- arrayOpsOperations)
      yield Array(a, b, s, f)

  def lift(f: (Int, Int) => Int): (Seq[Integer], Seq[Integer]) => Seq[Integer] = {
    case (a, b) =>
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) => Option(i).zip(Option(j)).headOption.map[Integer] { case (m, n) => f(m, n) }.orNull }
      }.orNull
  }

  @Test(dataProvider = "arrayOps")
  def arrayOps(a: Seq[Integer], b: Seq[Integer], s: String, f: (Int, Int) => Int) {
    assertEvalsTo(invoke(s, toIRArray(a), toIRArray(b)), lift(f)(a, b))
  }

  @Test(dataProvider = "arrayOpsOperations")
  def arrayOpsDifferentLength(s: String, f: (Int, Int) => Int) {
    assertFatal(invoke(s, IRArray(1, 2, 3), IRArray(1, 2)), "Arrays must have same length")
    assertFatal(invoke(s, IRArray(1, 2), IRArray(1, 2, 3)), "Arrays must have same length")
  }

  @Test def indexing() {
    val a = IRArray(0, null, 2)
    assertEvalsTo(invoke("[]", a, I32(0)), 0)
    assertEvalsTo(invoke("[]", a, I32(1)), null)
    assertEvalsTo(invoke("[]", a, I32(2)), 2)
    assertFatal(invoke("[]", a, I32(4)), "array index out of bounds")
    assertFatal(invoke("[]", a, I32(-1)), "array index out of bounds")
    assertEvalsTo(invoke("[]", naa, I32(2)), null)
    assertEvalsTo(invoke("[]", a, NA(TInt32())), null)
  }

  @Test def slicing() {
    val a = IRArray(0, null, 2)
    assertEvalsTo(invoke("[:]", a), IndexedSeq(0, null, 2))
    assertEvalsTo(invoke("[:]", naa), null)

    assertEvalsTo(invoke("[*:]", a, I32(1)), IndexedSeq(null, 2))
    assertEvalsTo(invoke("[*:]", a, I32(-2)), IndexedSeq(null, 2))
    assertEvalsTo(invoke("[*:]", a, I32(5)), IndexedSeq())
    assertEvalsTo(invoke("[*:]", a, I32(-5)), IndexedSeq(0, null, 2))
    assertEvalsTo(invoke("[*:]", naa, I32(1)), null)
    assertEvalsTo(invoke("[*:]", a, NA(TInt32())), null)

    assertEvalsTo(invoke("[:*]", a, I32(2)), IndexedSeq(0, null))
    assertEvalsTo(invoke("[:*]", a, I32(-1)), IndexedSeq(0, null))
    assertEvalsTo(invoke("[:*]", a, I32(5)), IndexedSeq(0, null, 2))
    assertEvalsTo(invoke("[:*]", a, I32(-5)), IndexedSeq())
    assertEvalsTo(invoke("[:*]", naa, I32(1)), null)
    assertEvalsTo(invoke("[:*]", a, NA(TInt32())), null)

    assertEvalsTo(invoke("[*:*]", a, I32(1), I32(3)), IndexedSeq(null, 2))
    assertEvalsTo(invoke("[*:*]", a, I32(1), I32(2)), IndexedSeq(null))
    assertEvalsTo(invoke("[*:*]", a, I32(0), I32(2)), IndexedSeq(0, null))
    assertEvalsTo(invoke("[*:*]", a, I32(0), I32(3)), IndexedSeq(0, null, 2))
    assertEvalsTo(invoke("[*:*]", a, I32(-1), I32(3)), IndexedSeq(2))
    assertEvalsTo(invoke("[*:*]", a, I32(-4), I32(4)), IndexedSeq(0, null, 2))
    assertEvalsTo(invoke("[*:*]", naa, I32(1), I32(2)), null)
    assertEvalsTo(invoke("[*:*]", a, I32(1), NA(TInt32())), null)
    assertEvalsTo(invoke("[*:*]", a, NA(TInt32()), I32(1)), null)
  }
}

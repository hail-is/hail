package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.types.{virtual, _}
import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import org.testng.annotations.{DataProvider, Test}
import org.scalatest.testng.TestNGSuite

class ArrayFunctionsSuite extends HailSuite {
  val naa = NA(TArray(TInt32))

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3, 7)),
    Array(null),
    Array(FastIndexedSeq(3, null, 7, null)),
    Array(FastIndexedSeq())
  )

  @DataProvider(name = "basicPairs")
  def basicPairsData(): Array[Array[Any]] = basicData().flatten.combinations(2).toArray

  @Test(dataProvider = "basic")
  def isEmpty(a: Seq[Integer]) {
    assertEvalsTo(invoke("isEmpty", TBoolean, toIRArray(a)),
      Option(a).map(_.isEmpty).orNull)
  }

  @Test(dataProvider = "basic")
  def append(a: Seq[Integer]) {
    assertEvalsTo(invoke("append", TArray(TInt32), toIRArray(a), I32(1)),
      Option(a).map(_ :+ 1).orNull)
  }

  @Test(dataProvider = "basic")
  def appendNull(a: Seq[Integer]) {
    assertEvalsTo(invoke("append", TArray(TInt32), toIRArray(a), NA(TInt32)),
      Option(a).map(_ :+ null).orNull)
  }

  @Test(dataProvider = "basic")
  def sum(a: Seq[Integer]) {
    assertEvalsTo(invoke("sum", TInt32, toIRArray(a)),
      Option(a).flatMap(_.foldLeft[Option[Int]](Some(0))((comb, x) => comb.flatMap(c => Option(x).map(_ + c)))).orNull)
  }

  @Test(dataProvider = "basic")
  def product(a: Seq[Integer]) {
    assertEvalsTo(invoke("product", TInt32, toIRArray(a)),
      Option(a).flatMap(_.foldLeft[Option[Int]](Some(1))((comb, x) => comb.flatMap(c => Option(x).map(_ * c)))).orNull)
  }

  @Test def mean() {
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7)), 5.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, null, 7)), null)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(3, 7, 11)), 7.0)
    assertEvalsTo(invoke("mean", TFloat64, IRArray()), Double.NaN)
    assertEvalsTo(invoke("mean", TFloat64, IRArray(null)), null)
    assertEvalsTo(invoke("mean", TFloat64, naa), null)
  }

  @Test def median() {
    assertEvalsTo(invoke("median", TInt32, IRArray(5)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(5, null, null)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, 7)), 5)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, null, 7, 1, null)), 3)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, 7, 1)), 3)
    assertEvalsTo(invoke("median", TInt32, IRArray(3, null, 9, 6, 1, null)), 4)
    assertEvalsTo(invoke("median", TInt32, IRArray()), null)
    assertEvalsTo(invoke("median", TInt32, IRArray(null)), null)
    assertEvalsTo(invoke("median", TInt32, naa), null)
  }
  
  @Test(dataProvider = "basicPairs")
  def extend(a: Seq[Integer], b: Seq[Integer]) {
    assertEvalsTo(invoke("extend", TArray(TInt32), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (x, y) => x ++ y}.orNull)
  }

  @DataProvider(name = "sort")
  def sortData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3, 9, 7), FastIndexedSeq(3, 7, 9), FastIndexedSeq(9, 7, 3)),
    Array(null, null, null),
    Array(FastIndexedSeq(3, null, 1, null, 3), FastIndexedSeq(1, 3, 3, null, null), FastIndexedSeq(3, 3, 1, null, null)),
    Array(FastIndexedSeq(1, null, 3, null, 1), FastIndexedSeq(1, 1, 3, null, null), FastIndexedSeq(3, 1, 1, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider = "sort")
  def min(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("min", TInt32, toIRArray(a)),
      Option(asc).filter(!_.contains(null)).flatMap(_.headOption).orNull)
  }

  @Test def testMinMaxNans() {
    assertAllEvalTo(
      (invoke("min", TFloat32, MakeArray(FastSeq(F32(Float.NaN), F32(1.0f), F32(Float.NaN), F32(111.0f)), TArray(TFloat32))), Float.NaN),
      (invoke("max", TFloat32, MakeArray(FastSeq(F32(Float.NaN), F32(1.0f), F32(Float.NaN), F32(111.0f)), TArray(TFloat32))), Float.NaN),
      (invoke("min", TFloat64, MakeArray(FastSeq(F64(Double.NaN), F64(1.0), F64(Double.NaN), F64(111.0)), TArray(TFloat64))), Double.NaN),
      (invoke("max", TFloat64, MakeArray(FastSeq(F64(Double.NaN), F64(1.0), F64(Double.NaN), F64(111.0)), TArray(TFloat64))), Double.NaN)
    )
  }

  @Test(dataProvider = "sort")
  def max(a: Seq[Integer], asc: Seq[Integer], desc: Seq[Integer]) {
    assertEvalsTo(invoke("max", TInt32, toIRArray(a)),
      Option(desc).filter(!_.contains(null)).flatMap(_.headOption).orNull)
  }

  @DataProvider(name = "argminmax")
  def argMinMaxData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(FastIndexedSeq(3, null, 1, null, 3), 2, 0),
    Array(FastIndexedSeq(1, null, 3, null, 1), 0, 2),
    Array(FastIndexedSeq(), null, null)
  )

  @Test(dataProvider = "argminmax")
  def argmin(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmin", TInt32, toIRArray(a)), argmin)
  }

  @Test(dataProvider = "argminmax")
  def argmax(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmax", TInt32, toIRArray(a)), argmax)
  }

  @DataProvider(name = "uniqueMinMaxIndex")
  def uniqueMinMaxData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(FastIndexedSeq(3, null, 1, null, 3), 2, null),
    Array(FastIndexedSeq(1, null, 3, null, 1), null, 2),
    Array(FastIndexedSeq(), null, null)
  )

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMinIndex(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMinIndex", TInt32, toIRArray(a)), argmin)
  }

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMaxIndex(a: Seq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMaxIndex", TInt32, toIRArray(a)), argmax)
  }

  @DataProvider(name = "arrayOpsData")
  def arrayOpsData(): Array[Array[Any]] = Array[Any](
    FastIndexedSeq(3, 9, 7, 1),
    FastIndexedSeq(null, 2, null, 8),
    FastIndexedSeq(5, 3, null, null),
    null
  ).combinations(2).toArray

  @DataProvider(name = "arrayOpsOperations")
  def arrayOpsOperations: Array[Array[Any]] = Array[(String, (Int, Int) => Int)](
    ("add", _ + _),
    ("sub", _ - _),
    ("mul", _ * _),
    ("floordiv", _ / _),
    ("mod", _ % _)
  ).map(_.productIterator.toArray)

  @DataProvider(name = "arrayOps")
  def arrayOpsPairs(): Array[Array[Any]] =
    for (Array(a, b) <- arrayOpsData(); Array(s, f) <- arrayOpsOperations)
      yield Array(a, b, s, f)

  def lift(f: (Int, Int) => Int): (IndexedSeq[Integer], IndexedSeq[Integer]) => IndexedSeq[Integer] = {
    case (a, b) =>
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) => Option(i).zip(Option(j)).headOption.map[Integer] { case (m, n) => f(m, n) }.orNull }
      }.orNull
  }

  @Test(dataProvider = "arrayOps")
  def arrayOps(a: IndexedSeq[Integer], b: IndexedSeq[Integer], s: String, f: (Int, Int) => Int) {
    assertEvalsTo(invoke(s, TArray(TInt32), toIRArray(a), toIRArray(b)), lift(f)(a, b))
  }

  @Test(dataProvider = "arrayOpsData")
  def arrayOpsFPDiv(a: IndexedSeq[Integer], b: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("div", TArray(TFloat32), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) => Option(i).zip(Option(j)).headOption.map[java.lang.Float] { case (m, n) => m.toFloat / n }.orNull }
      }.orNull )
  }

  @Test(dataProvider = "arrayOpsData")
  def arrayOpsPow(a: IndexedSeq[Integer], b: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("pow", TArray(TFloat64), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) => Option(i).zip(Option(j)).headOption.map[java.lang.Double] { case (m, n) => math.pow(m.toDouble, n.toDouble) }.orNull }
      }.orNull )
  }

  @Test(dataProvider = "arrayOpsOperations")
  def arrayOpsDifferentLength(s: String, f: (Int, Int) => Int) {
    assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2, 3), IRArray(1, 2)), "length mismatch")
    assertFatal(invoke(s, TArray(TInt32), IRArray(1, 2), IRArray(1, 2, 3)), "length mismatch")
  }

  @Test def indexing() {
    val a = IRArray(0, null, 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(0), Str("")), 0)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(1), Str("")), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(2), Str("")), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-1), Str("")), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-3), Str("")), 0)
    assertFatal(invoke("indexArray", TInt32, a, I32(3), Str("")), "array index out of bounds")
    assertFatal(invoke("indexArray", TInt32, a, I32(-4), Str("")), "array index out of bounds")
    assertEvalsTo(invoke("indexArray", TInt32, naa, I32(2), Str("")), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, NA(TInt32), Str("")), null)
  }

  @Test def slicing() {
    val a = IRArray(0, null, 2)
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), a, I32(1)), FastIndexedSeq(null, 2))
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), a, I32(-2)), FastIndexedSeq(null, 2))
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), a, I32(5)), FastIndexedSeq())
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), a, I32(-5)), FastIndexedSeq(0, null, 2))
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), naa, I32(1)), null)
    assertEvalsTo(invoke("sliceRight", TArray(TInt32), a, NA(TInt32)), null)

    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), a, I32(2)), FastIndexedSeq(0, null))
    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), a, I32(-1)), FastIndexedSeq(0, null))
    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), a, I32(5)), FastIndexedSeq(0, null, 2))
    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), a, I32(-5)), FastIndexedSeq())
    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), naa, I32(1)), null)
    assertEvalsTo(invoke("sliceLeft", TArray(TInt32), a, NA(TInt32)), null)

    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(1), I32(3)), FastIndexedSeq(null, 2))
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(1), I32(2)), FastIndexedSeq(null))
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(0), I32(2)), FastIndexedSeq(0, null))
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(0), I32(3)), FastIndexedSeq(0, null, 2))
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(-1), I32(3)), FastIndexedSeq(2))
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(-4), I32(4)), FastIndexedSeq(0, null, 2))
    assertEvalsTo(invoke("slice", TArray(TInt32), naa, I32(1), I32(2)), null)
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(1), NA(TInt32)), null)
    assertEvalsTo(invoke("slice", TArray(TInt32), a, NA(TInt32), I32(1)), null)
    assertEvalsTo(invoke("slice", TArray(TInt32), a, I32(3), I32(2)), FastIndexedSeq())
  }

  @DataProvider(name = "flatten")
  def flattenData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(FastIndexedSeq(3, 9, 7), FastIndexedSeq(3, 7, 9)), FastIndexedSeq(3, 9, 7, 3, 7, 9)),
    Array(FastIndexedSeq(null, FastIndexedSeq(1)), FastIndexedSeq(1)),
    Array(FastIndexedSeq(null, null), FastIndexedSeq()),
    Array(FastIndexedSeq(FastIndexedSeq(null), FastIndexedSeq(), FastIndexedSeq(7)), FastIndexedSeq(null, 7)),
    Array(FastIndexedSeq(FastIndexedSeq(), FastIndexedSeq()), FastIndexedSeq())
  )

  @Test(dataProvider = "flatten")
  def flatten(in: IndexedSeq[IndexedSeq[Integer]], expected: IndexedSeq[Int]) {
    assertEvalsTo(invoke("flatten", TArray(TInt32), MakeArray(in.map(toIRArray(_)), TArray(TArray(TInt32)))), expected)
  }

  @Test def testContains() {
    val t = TArray(TString)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastIndexedSeq(FastIndexedSeq() -> t),
      expected=false)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastIndexedSeq(FastIndexedSeq(null) -> t),
      expected=false)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastIndexedSeq(FastIndexedSeq("c", "a", "b") -> t),
      expected=true)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastIndexedSeq(FastIndexedSeq("c", "a", "b", null) -> t),
      expected=true)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastIndexedSeq((null, t)),
      expected=null)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastIndexedSeq((null, t)),
      expected=null)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastIndexedSeq(FastIndexedSeq("a", null) -> t),
      expected=true)
  }
}

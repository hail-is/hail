package is.hail.expr.ir

import is.hail.TestUtils._
import is.hail.expr.ir.TestUtils._
import is.hail.types.virtual._
import is.hail.utils.FastSeq
import is.hail.{ExecStrategy, HailSuite}
import org.testng.annotations.{DataProvider, Test}

class ArrayFunctionsSuite extends HailSuite {
  val naa = NA(TArray(TInt32))

  implicit val execStrats = ExecStrategy.javaOnly

  @DataProvider(name = "basic")
  def basicData(): Array[Array[Any]] = Array(
    Array(FastSeq(3, 7)),
    Array(null),
    Array(FastSeq(3, null, 7, null)),
    Array(FastSeq())
  )

  @DataProvider(name = "basicPairs")
  def basicPairsData(): Array[Array[Any]] = basicData().flatten.combinations(2).toArray

  @Test(dataProvider = "basic")
  def isEmpty(a: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("isEmpty", TBoolean, toIRArray(a)),
      Option(a).map(_.isEmpty).orNull)
  }

  @Test(dataProvider = "basic")
  def append(a: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("append", TArray(TInt32), toIRArray(a), I32(1)),
      Option(a).map(_ :+ 1).orNull)
  }

  @Test(dataProvider = "basic")
  def appendNull(a: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("append", TArray(TInt32), toIRArray(a), NA(TInt32)),
      Option(a).map(_ :+ null).orNull)
  }

  @Test(dataProvider = "basic")
  def sum(a: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("sum", TInt32, toIRArray(a)),
      Option(a).flatMap(_.foldLeft[Option[Int]](Some(0))((comb, x) => comb.flatMap(c => Option(x).map(_ + c)))).orNull)
  }

  @Test(dataProvider = "basic")
  def product(a: IndexedSeq[Integer]) {
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
  def extend(a: IndexedSeq[Integer], b: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("extend", TArray(TInt32), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (x, y) => x ++ y}.orNull)
  }

  @DataProvider(name = "sort")
  def sortData(): Array[Array[Any]] = Array(
    Array(FastSeq(3, 9, 7), FastSeq(3, 7, 9), FastSeq(9, 7, 3)),
    Array(null, null, null),
    Array(FastSeq(3, null, 1, null, 3), FastSeq(1, 3, 3, null, null), FastSeq(3, 3, 1, null, null)),
    Array(FastSeq(1, null, 3, null, 1), FastSeq(1, 1, 3, null, null), FastSeq(3, 1, 1, null, null)),
    Array(FastSeq(), FastSeq(), FastSeq())
  )

  @Test(dataProvider = "sort")
  def min(a: IndexedSeq[Integer], asc: IndexedSeq[Integer], desc: IndexedSeq[Integer]) {
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
  def max(a: IndexedSeq[Integer], asc: IndexedSeq[Integer], desc: IndexedSeq[Integer]) {
    assertEvalsTo(invoke("max", TInt32, toIRArray(a)),
      Option(desc).filter(!_.contains(null)).flatMap(_.headOption).orNull)
  }

  @DataProvider(name = "argminmax")
  def argMinMaxData(): Array[Array[Any]] = Array(
    Array(FastSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(FastSeq(3, null, 1, null, 3), 2, 0),
    Array(FastSeq(1, null, 3, null, 1), 0, 2),
    Array(FastSeq(), null, null)
  )

  @Test(dataProvider = "argminmax")
  def argmin(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmin", TInt32, toIRArray(a)), argmin)
  }

  @Test(dataProvider = "argminmax")
  def argmax(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("argmax", TInt32, toIRArray(a)), argmax)
  }

  @DataProvider(name = "uniqueMinMaxIndex")
  def uniqueMinMaxData(): Array[Array[Any]] = Array(
    Array(FastSeq(3, 9, 7), 0, 1),
    Array(null, null, null),
    Array(FastSeq(3, null, 1, null, 3), 2, null),
    Array(FastSeq(1, null, 3, null, 1), null, 2),
    Array(FastSeq(), null, null)
  )

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMinIndex(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMinIndex", TInt32, toIRArray(a)), argmin)
  }

  @Test(dataProvider = "uniqueMinMaxIndex")
  def uniqueMaxIndex(a: IndexedSeq[Integer], argmin: Integer, argmax: Integer) {
    assertEvalsTo(invoke("uniqueMaxIndex", TInt32, toIRArray(a)), argmax)
  }

  @DataProvider(name = "arrayOpsData")
  def arrayOpsData(): Array[Array[Any]] = Array[Any](
    FastSeq(3, 9, 7, 1),
    FastSeq(null, 2, null, 8),
    FastSeq(5, 3, null, null),
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
    assertEvalsTo(invoke("div", TArray(TFloat64), toIRArray(a), toIRArray(b)),
      Option(a).zip(Option(b)).headOption.map { case (a0, b0) =>
        a0.zip(b0).map { case (i, j) => Option(i).zip(Option(j)).headOption.map[java.lang.Double] { case (m, n) => m.toDouble / n }.orNull }
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
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(0)), 0)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(1)), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(2)), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-1)), 2)
    assertEvalsTo(invoke("indexArray", TInt32, a, I32(-3)), 0)
    assertFatal(invoke("indexArray", TInt32, a, I32(3)), "array index out of bounds")
    assertFatal(invoke("indexArray", TInt32, a, I32(-4)), "array index out of bounds")
    assertEvalsTo(invoke("indexArray", TInt32, naa, I32(2)), null)
    assertEvalsTo(invoke("indexArray", TInt32, a, NA(TInt32)), null)
  }

  @Test def slicing() {
    val a = IRArray(0, null, 2)
    assertEvalsTo(ArraySlice(a, I32(1), None), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(-2), None), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(5), None), FastSeq())
    assertEvalsTo(ArraySlice(a, I32(-5), None), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(naa, I32(1), None), null)
    assertEvalsTo(ArraySlice(a, NA(TInt32), None), null)

    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(2))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(-1))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(5))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(-5))), FastSeq())
    assertEvalsTo(ArraySlice(naa, I32(0), Some(I32(1))), null)
    assertEvalsTo(ArraySlice(a, I32(0), Some(NA(TInt32))), null)

    assertEvalsTo(ArraySlice(a, I32(1), Some(I32(3))), FastSeq(null, 2))
    assertEvalsTo(ArraySlice(a, I32(1), Some(I32(2))), FastSeq(null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(2))), FastSeq(0, null))
    assertEvalsTo(ArraySlice(a, I32(0), Some(I32(3))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(a, I32(-1),Some( I32(3))), FastSeq(2))
    assertEvalsTo(ArraySlice(a, I32(-4), Some(I32(4))), FastSeq(0, null, 2))
    assertEvalsTo(ArraySlice(naa, I32(1), Some(I32(2))), null)
    assertEvalsTo(ArraySlice(a, I32(1), Some(NA(TInt32))), null)
    assertEvalsTo(ArraySlice(a, NA(TInt32), Some(I32(1))), null)
    assertEvalsTo(ArraySlice(a, I32(3), Some(I32(2))), FastSeq())
  }

  @DataProvider(name = "flatten")
  def flattenData(): Array[Array[Any]] = Array(
    Array(FastSeq(FastSeq(3, 9, 7), FastSeq(3, 7, 9)), FastSeq(3, 9, 7, 3, 7, 9)),
    Array(FastSeq(null, FastSeq(1)), FastSeq(1)),
    Array(FastSeq(null, null), FastSeq()),
    Array(FastSeq(FastSeq(null), FastSeq(), FastSeq(7)), FastSeq(null, 7)),
    Array(FastSeq(FastSeq(), FastSeq()), FastSeq())
  )

  @Test(dataProvider = "flatten")
  def flatten(in: IndexedSeq[IndexedSeq[Integer]], expected: IndexedSeq[Int]) {
    assertEvalsTo(invoke("flatten", TArray(TInt32), MakeArray(in.map(toIRArray(_)), TArray(TArray(TInt32)))), expected)
  }

  @Test def testContains() {
    val t = TArray(TString)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq() -> t),
      expected=false)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq(null) -> t),
      expected=false)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq("c", "a", "b") -> t),
      expected=true)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq(FastSeq("c", "a", "b", null) -> t),
      expected=true)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), Str("a")),
      args = FastSeq((null, t)),
      expected=null)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastSeq((null, t)),
      expected=null)

    assertEvalsTo(
      invoke("contains", TBoolean, In(0, t), NA(t.elementType)),
      args = FastSeq(FastSeq("a", null) -> t),
      expected=true)
  }
}

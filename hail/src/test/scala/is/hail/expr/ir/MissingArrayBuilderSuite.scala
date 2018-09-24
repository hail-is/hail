package is.hail.expr.ir

import is.hail.asm4s.AsmFunction2
import is.hail.utils.FastIndexedSeq
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

import scala.reflect.ClassTag

class MissingArrayBuilderSuite extends TestNGSuite {
  def ordering[T <: AnyVal](f: (T, T) => Boolean): AsmFunction2[T, T, Boolean] =
    new AsmFunction2[T, T, Boolean] {
      def apply(i: T, j: T): Boolean = f(i, j)
    }

  def addToArrayBuilder[B <: MissingArrayBuilder, T](
    ab: B, array: IndexedSeq[T]
  )(f: (B, T) => Unit): Unit = {
    array.foreach { i =>
      if (i == null)
        ab.addMissing()
      else
        f(ab, i)
    }
  }

  def getResult[B <: MissingArrayBuilder, T >: Null : ClassTag](ab: B)(f: (B, Int) => T): Array[T] = {
    Array.tabulate[T](ab.size) { i =>
      if (ab.isMissing(i))
        null
      else
        f(ab, i)
    }
  }

  @DataProvider(name = "sortInt")
  def integerData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3, null, 3, 7, null), FastIndexedSeq(3, 3, 7, null, null)),
    Array(FastIndexedSeq(null, null, null, null), FastIndexedSeq(null, null, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider="sortInt")
  def testSortOnIntArrayBuilder(array: IndexedSeq[Integer], expected: IndexedSeq[Integer]) {
    val ab = new IntArrayBuilder(16)
    addToArrayBuilder(ab, array) { (iab, i) => iab.add(i) }

    ab.sort(ordering[Int] { (i, j) => i < j})
    val result = getResult[IntArrayBuilder, Integer](ab) { (iab, i) => Int.box(iab(i)) }
    assert(result sameElements expected)
  }

  @DataProvider(name = "sortLong")
  def longData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3L, null, 3L, 7L, null), FastIndexedSeq(3L, 3L, 7L, null, null)),
    Array(FastIndexedSeq(null, null, null, null), FastIndexedSeq(null, null, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider="sortLong")
  def testSortOnLongArrayBuilder(array: IndexedSeq[java.lang.Long], expected: IndexedSeq[java.lang.Long]) {
    val ab = new LongArrayBuilder(16)
    addToArrayBuilder(ab, array) { (jab, j) => jab.add(j) }

    ab.sort(ordering[Long] { (i, j) => i < j})
    val result = getResult[LongArrayBuilder, java.lang.Long](ab) { (jab, j) => Long.box(jab(j)) }
    assert(result sameElements expected)
  }

  @DataProvider(name = "sortFloat")
  def floatData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3f, null, 3f, 7f, null), FastIndexedSeq(3f, 3f, 7f, null, null)),
    Array(FastIndexedSeq(null, null, null, null), FastIndexedSeq(null, null, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider="sortFloat")
  def testSortOnFloatArrayBuilder(array: IndexedSeq[java.lang.Float], expected: IndexedSeq[java.lang.Float]) {
    val ab = new FloatArrayBuilder(16)
    addToArrayBuilder(ab, array) { (fab, f) => fab.add(f) }

    ab.sort(ordering[Float] { (i, j) => i < j })
    val result = getResult[FloatArrayBuilder, java.lang.Float](ab) { (fab, f) => Float.box(fab(f)) }
    assert(result sameElements expected)
  }

  @DataProvider(name = "sortDouble")
  def doubleData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(3d, null, 3d, 7d, null), FastIndexedSeq(3d, 3d, 7d, null, null)),
    Array(FastIndexedSeq(null, null, null, null), FastIndexedSeq(null, null, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider="sortDouble")
  def testSortOnDoubleArrayBuilder(array: IndexedSeq[java.lang.Double], expected: IndexedSeq[java.lang.Double]) {
    val ab = new DoubleArrayBuilder(16)
    addToArrayBuilder(ab, array) { (dab, d) => dab.add(d) }

    ab.sort(ordering[Double] { (i, j) => i < j })
    val result = getResult[DoubleArrayBuilder, java.lang.Double](ab) { (dab, d) => Double.box(dab(d)) }
    assert(result sameElements expected)
  }

  @DataProvider(name = "sortBoolean")
  def booleanData(): Array[Array[Any]] = Array(
    Array(FastIndexedSeq(true, null, true, false, null), FastIndexedSeq(false, true, true, null, null)),
    Array(FastIndexedSeq(null, null, null, null), FastIndexedSeq(null, null, null, null)),
    Array(FastIndexedSeq(), FastIndexedSeq())
  )

  @Test(dataProvider="sortBoolean")
  def testSortOnBooleanArrayBuilder(array: IndexedSeq[java.lang.Boolean], expected: IndexedSeq[java.lang.Boolean]) {
    val ab = new BooleanArrayBuilder(16)
    addToArrayBuilder(ab, array) { (bab, b) => bab.add(b) }

    ab.sort(ordering[Boolean] { (i, j) => i < j })
    val result = getResult[BooleanArrayBuilder, java.lang.Boolean](ab) { (bab, b) => Boolean.box(bab(b)) }
    assert(result sameElements expected)
  }
}

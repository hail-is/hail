package is.hail.expr.ir

import is.hail.ParameterizedTest
import is.hail.asm4s.AsmFunction2
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq

import scala.reflect.ClassTag

class MissingArrayBuilderSuite {
  def ordering[T <: AnyVal](f: (T, T) => Boolean): AsmFunction2[T, T, Boolean] =
    new AsmFunction2[T, T, Boolean] {
      override def apply(i: T, j: T): Boolean = f(i, j)
    }

  def addToArrayBuilder[B <: MissingArrayBuilder, T](
    ab: B,
    array: IndexedSeq[T],
  )(
    f: (B, T) => Unit
  ): Unit = {
    array.foreach { i =>
      if (i == null)
        ab.addMissing()
      else
        f(ab, i)
    }
  }

  def getResult[B <: MissingArrayBuilder, T >: Null: ClassTag](ab: B)(f: (B, Int) => T)
    : Array[T] = {
    Array.tabulate[T](ab.size) { i =>
      if (ab.isMissing(i))
        null
      else
        f(ab, i)
    }
  }

  def testSortOnIntArrayBuilder() = ArraySeq[(IndexedSeq[Integer], IndexedSeq[Integer])](
    (FastSeq(3, null, 3, 7, null), FastSeq(3, 3, 7, null, null)),
    (FastSeq(null, null, null, null), FastSeq(null, null, null, null)),
    (FastSeq(), FastSeq()),
  )

  @ParameterizedTest
  def testSortOnIntArrayBuilder(array: IndexedSeq[Integer], expected: IndexedSeq[Integer]): Unit = {
    val ab = new IntMissingArrayBuilder(16)
    addToArrayBuilder(ab, array)((iab, i) => iab.add(i))

    ab.sort(ordering[Int]((i, j) => i < j))
    val result = getResult[IntMissingArrayBuilder, Integer](ab)((iab, i) => Int.box(iab(i)))
    assert(result sameElements expected)
  }

  def testSortOnLongArrayBuilder() =
    ArraySeq[(IndexedSeq[java.lang.Long], IndexedSeq[java.lang.Long])](
      (FastSeq(3L, null, 3L, 7L, null), FastSeq(3L, 3L, 7L, null, null)),
      (FastSeq(null, null, null, null), FastSeq(null, null, null, null)),
      (FastSeq(), FastSeq()),
    )

  @ParameterizedTest
  def testSortOnLongArrayBuilder(
    array: IndexedSeq[java.lang.Long],
    expected: IndexedSeq[java.lang.Long],
  ): Unit = {
    val ab = new LongMissingArrayBuilder(16)
    addToArrayBuilder(ab, array)((jab, j) => jab.add(j))

    ab.sort(ordering[Long]((i, j) => i < j))
    val result = getResult[LongMissingArrayBuilder, java.lang.Long](ab) { (jab, j) =>
      Long.box(jab(j))
    }
    assert(result sameElements expected)
  }

  def testSortOnFloatArrayBuilder() =
    ArraySeq[(IndexedSeq[java.lang.Float], IndexedSeq[java.lang.Float])](
      (FastSeq(3f, null, 3f, 7f, null), FastSeq(3f, 3f, 7f, null, null)),
      (FastSeq(null, null, null, null), FastSeq(null, null, null, null)),
      (FastSeq(), FastSeq()),
    )

  @ParameterizedTest
  def testSortOnFloatArrayBuilder(
    array: IndexedSeq[java.lang.Float],
    expected: IndexedSeq[java.lang.Float],
  ): Unit = {
    val ab = new FloatMissingArrayBuilder(16)
    addToArrayBuilder(ab, array)((fab, f) => fab.add(f))

    ab.sort(ordering[Float]((i, j) => i < j))
    val result = getResult[FloatMissingArrayBuilder, java.lang.Float](ab) { (fab, f) =>
      Float.box(fab(f))
    }
    assert(result sameElements expected)
  }

  def testSortOnDoubleArrayBuilder() =
    ArraySeq[(IndexedSeq[java.lang.Double], IndexedSeq[java.lang.Double])](
      (FastSeq(3d, null, 3d, 7d, null), FastSeq(3d, 3d, 7d, null, null)),
      (FastSeq(null, null, null, null), FastSeq(null, null, null, null)),
      (FastSeq(), FastSeq()),
    )

  @ParameterizedTest
  def testSortOnDoubleArrayBuilder(
    array: IndexedSeq[java.lang.Double],
    expected: IndexedSeq[java.lang.Double],
  ): Unit = {
    val ab = new DoubleMissingArrayBuilder(16)
    addToArrayBuilder(ab, array)((dab, d) => dab.add(d))

    ab.sort(ordering[Double]((i, j) => i < j))
    val result = getResult[DoubleMissingArrayBuilder, java.lang.Double](ab) { (dab, d) =>
      Double.box(dab(d))
    }
    assert(result sameElements expected)
  }

  def testSortOnBooleanArrayBuilder() =
    ArraySeq[(IndexedSeq[java.lang.Boolean], IndexedSeq[java.lang.Boolean])](
      (FastSeq(true, null, true, false, null), FastSeq(false, true, true, null, null)),
      (FastSeq(null, null, null, null), FastSeq(null, null, null, null)),
      (FastSeq(), FastSeq()),
    )

  @ParameterizedTest
  def testSortOnBooleanArrayBuilder(
    array: IndexedSeq[java.lang.Boolean],
    expected: IndexedSeq[java.lang.Boolean],
  ): Unit = {
    val ab = new BooleanMissingArrayBuilder(16)
    addToArrayBuilder(ab, array)((bab, b) => bab.add(b))

    ab.sort(ordering[Boolean]((i, j) => i < j))
    val result = getResult[BooleanMissingArrayBuilder, java.lang.Boolean](ab) { (bab, b) =>
      Boolean.box(bab(b))
    }
    assert(result sameElements expected)
  }
}

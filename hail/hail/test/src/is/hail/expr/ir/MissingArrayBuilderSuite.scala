package is.hail.expr.ir

import is.hail.TestCaseSupport
import is.hail.asm4s.AsmFunction2
import is.hail.collection.FastSeq

import scala.reflect.ClassTag

class MissingArrayBuilderSuite extends munit.FunSuite with TestCaseSupport {
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

  object checkSortOnIntArrayBuilder extends TestCases {
    def apply(
      array: IndexedSeq[Integer],
      expected: IndexedSeq[Integer],
    )(implicit loc: munit.Location
    ): Unit = test("sort int array builder") {
      val ab = new IntMissingArrayBuilder(16)
      addToArrayBuilder(ab, array)((iab, i) => iab.add(i))
      ab.sort(ordering[Int]((i, j) => i < j))
      val result = getResult[IntMissingArrayBuilder, Integer](ab)((iab, i) => Int.box(iab(i)))
      assert(result sameElements expected)
    }
  }

  checkSortOnIntArrayBuilder(FastSeq(3, null, 3, 7, null), FastSeq(3, 3, 7, null, null))
  checkSortOnIntArrayBuilder(FastSeq(null, null, null, null), FastSeq(null, null, null, null))
  checkSortOnIntArrayBuilder(FastSeq(), FastSeq())

  object checkSortOnLongArrayBuilder extends TestCases {
    def apply(
      array: IndexedSeq[java.lang.Long],
      expected: IndexedSeq[java.lang.Long],
    )(implicit loc: munit.Location
    ): Unit = test("sort long array builder") {
      val ab = new LongMissingArrayBuilder(16)
      addToArrayBuilder(ab, array)((jab, j) => jab.add(j))
      ab.sort(ordering[Long]((i, j) => i < j))
      val result = getResult[LongMissingArrayBuilder, java.lang.Long](ab) { (jab, j) =>
        Long.box(jab(j))
      }
      assert(result sameElements expected)
    }
  }

  checkSortOnLongArrayBuilder(FastSeq(3L, null, 3L, 7L, null), FastSeq(3L, 3L, 7L, null, null))
  checkSortOnLongArrayBuilder(FastSeq(null, null, null, null), FastSeq(null, null, null, null))
  checkSortOnLongArrayBuilder(FastSeq(), FastSeq())

  object checkSortOnFloatArrayBuilder extends TestCases {
    def apply(
      array: IndexedSeq[java.lang.Float],
      expected: IndexedSeq[java.lang.Float],
    )(implicit loc: munit.Location
    ): Unit = test("sort float array builder") {
      val ab = new FloatMissingArrayBuilder(16)
      addToArrayBuilder(ab, array)((fab, f) => fab.add(f))
      ab.sort(ordering[Float]((i, j) => i < j))
      val result = getResult[FloatMissingArrayBuilder, java.lang.Float](ab) { (fab, f) =>
        Float.box(fab(f))
      }
      assert(result sameElements expected)
    }
  }

  checkSortOnFloatArrayBuilder(FastSeq(3f, null, 3f, 7f, null), FastSeq(3f, 3f, 7f, null, null))
  checkSortOnFloatArrayBuilder(FastSeq(null, null, null, null), FastSeq(null, null, null, null))
  checkSortOnFloatArrayBuilder(FastSeq(), FastSeq())

  object checkSortOnDoubleArrayBuilder extends TestCases {
    def apply(
      array: IndexedSeq[java.lang.Double],
      expected: IndexedSeq[java.lang.Double],
    )(implicit loc: munit.Location
    ): Unit = test("sort double array builder") {
      val ab = new DoubleMissingArrayBuilder(16)
      addToArrayBuilder(ab, array)((dab, d) => dab.add(d))
      ab.sort(ordering[Double]((i, j) => i < j))
      val result = getResult[DoubleMissingArrayBuilder, java.lang.Double](ab) { (dab, d) =>
        Double.box(dab(d))
      }
      assert(result sameElements expected)
    }
  }

  checkSortOnDoubleArrayBuilder(FastSeq(3d, null, 3d, 7d, null), FastSeq(3d, 3d, 7d, null, null))
  checkSortOnDoubleArrayBuilder(FastSeq(null, null, null, null), FastSeq(null, null, null, null))
  checkSortOnDoubleArrayBuilder(FastSeq(), FastSeq())

  object checkSortOnBooleanArrayBuilder extends TestCases {
    def apply(
      array: IndexedSeq[java.lang.Boolean],
      expected: IndexedSeq[java.lang.Boolean],
    )(implicit loc: munit.Location
    ): Unit = test("sort boolean array builder") {
      val ab = new BooleanMissingArrayBuilder(16)
      addToArrayBuilder(ab, array)((bab, b) => bab.add(b))
      ab.sort(ordering[Boolean]((i, j) => i < j))
      val result = getResult[BooleanMissingArrayBuilder, java.lang.Boolean](ab) { (bab, b) =>
        Boolean.box(bab(b))
      }
      assert(result sameElements expected)
    }
  }

  checkSortOnBooleanArrayBuilder(
    FastSeq(true, null, true, false, null),
    FastSeq(false, true, true, null, null),
  )

  checkSortOnBooleanArrayBuilder(FastSeq(null, null, null, null), FastSeq(null, null, null, null))
  checkSortOnBooleanArrayBuilder(FastSeq(), FastSeq())
}

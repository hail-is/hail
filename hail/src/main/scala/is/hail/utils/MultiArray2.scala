package is.hail.utils

import scala.collection.immutable.IndexedSeq
import scala.reflect.ClassTag

import java.io.Serializable

class MultiArray2[@specialized(Int, Long, Float, Double, Boolean) T](
  val n1: Int,
  val n2: Int,
  val a: Array[T],
) extends Serializable with Iterable[T] {

  require(n1 >= 0 && n2 >= 0)
  require(a.length == n1 * n2)

  class Row(val i: Int) extends IndexedSeq[T] {
    require(i >= 0 && i < n1)

    def apply(j: Int): T = {
      if (j < 0 || j >= length) throw new ArrayIndexOutOfBoundsException
      a(i * n2 + j)
    }

    def length: Int = n2
  }

  class Column(val j: Int) extends IndexedSeq[T] {
    require(j >= 0 && j < n2)

    def apply(i: Int): T = {
      if (i < 0 || i >= length) throw new ArrayIndexOutOfBoundsException
      a(i * n2 + j)
    }

    def length: Int = n1
  }

  def row(i: Int) = new Row(i)
  def column(j: Int) = new Column(j)

  def rows: Iterable[Row] = for (i <- rowIndices) yield row(i)
  def columns: Iterable[Column] = for (j <- columnIndices) yield column(j)

  def indices: Iterable[(Int, Int)] = for {
    i <- 0 until n1
    j <- 0 until n2
  } yield (i, j)

  def rowIndices: Iterable[Int] = 0 until n1

  def columnIndices: Iterable[Int] = 0 until n2

  def apply(i: Int, j: Int): T = {
    require(i >= 0 && i < n1 && j >= 0 && j < n2)
    a(i * n2 + j)
  }

  def update(i: Int, j: Int, x: T): Unit = {
    require(i >= 0 && i < n1 && j >= 0 && j < n2)
    a.update(i * n2 + j, x)
  }

  def update(t: (Int, Int), x: T): Unit = {
    require(t._1 >= 0 && t._1 < n1 && t._2 >= 0 && t._2 < n2)
    update(t._1, t._2, x)
  }

  def array: Array[T] = a

  def zip[S](other: MultiArray2[S]): MultiArray2[(T, S)] = {
    require(n1 == other.n1 && n2 == other.n2)
    new MultiArray2(n1, n2, a.zip(other.a))
  }

  def iterator: Iterator[T] = a.iterator
}

object MultiArray2 {
  def fill[T](n1: Int, n2: Int)(elem: => T)(implicit tct: ClassTag[T]): MultiArray2[T] =
    new MultiArray2[T](n1, n2, Array.fill[T](n1 * n2)(elem))

  def empty[T](implicit tct: ClassTag[T]): MultiArray2[T] =
    new MultiArray2[T](0, 0, Array.empty[T](tct))
}

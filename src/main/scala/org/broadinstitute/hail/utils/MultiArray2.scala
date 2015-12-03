package org.broadinstitute.hail.utils

import java.io.Serializable
import scala.reflect.ClassTag
import scala.collection.immutable.IndexedSeq


class MultiArray2[T](val n1: Int,
                     val n2: Int,
                     val a: Array[T]) extends Serializable with Iterable[T] {

  require(n1 >= 0 && n2 >= 0)
  require(a.length == n1*n2)

  class RowSlice(val i:Int) extends IndexedSeq[T] {
    require(i >= 0 && i < n1)
    def apply(j:Int): T = {
      a(i*n2 + j)
    }
    def length: Int = n2
  }

  class ColumnSlice(val j:Int) extends IndexedSeq[T] {
    require(j >= 0 && j < n2)
    def apply(i:Int): T = {
      a(i*n2 + j)
    }
    def length: Int = n1
  }

  def rowSlice(i:Int) = new RowSlice(i)
  def columnSlice(j:Int) = new ColumnSlice(j)

  def rows: Iterable[RowSlice] = for (i <- rowIndices) yield rowSlice(i)
  def columns: Iterable[ColumnSlice] = for (j <- columnIndices) yield columnSlice(j)

  def indices: Iterable[(Int,Int)] = for (i <- 0 until n1; j <- 0 until n2) yield (i, j)

  def rowIndices: Iterable[Int] = for (i <- 0 until n1) yield i

  def columnIndices: Iterable[Int] = for (j <- 0 until n2) yield j

  def apply(i: Int, j: Int): T = {
    require(i >= 0 && i < n1 && j >= 0 && j < n2)
    a(i*n2 + j)
  }

  def update(i: Int, j: Int, x:T): Unit = {
    require(i >= 0 && i < n1 && j >= 0 && j < n2)
    a.update(i*n2 + j,x)
  }

  def update(t: (Int,Int), x:T): Unit = {
    require(t._1 >= 0 && t._1 < n1 && t._2 >= 0 && t._2 < n2)
    update(t._1,t._2,x)
  }

  def toArray: Array[T] = a

  def zip[S](other: MultiArray2[S]): MultiArray2[(T,S)] = {
    require(n1 == other.n1 && n2 == other.n2)
    new MultiArray2(n1,n2,a.zip(other.a))
  }

  def iterator: Iterator[T] = a.iterator
}

object MultiArray2 {
  def fill[T](n1: Int, n2: Int)(elem: => T)(implicit tct: ClassTag[T]): MultiArray2[T] =
    new MultiArray2[T](n1, n2, Array.fill[T](n1 * n2)(elem))
}


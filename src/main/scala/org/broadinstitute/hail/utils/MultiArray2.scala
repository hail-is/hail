package org.broadinstitute.hail.utils

import java.io.Serializable
import scala.reflect.ClassTag


class MultiArray2[T](val n1: Int,
                     val n2: Int,
                     val a: Array[T]) extends Serializable {
  require(a.length == n1*n2)

  class rowSlice(val i:Int) {
    def indices: Iterable[(Int,Int)] = for (j <- 0 until n2) yield (i,j)
  }

  class columnSlice(val j:Int) {
    def indices: Iterable[(Int,Int)] = for (i <- 0 until n1) yield (i,j)
  }

  def rowSlice(i:Int): Array[T] = a.slice(i*n2 + 0, i*n2 + 3)

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

  def rows: Iterable[rowSlice] = for (i <- rowIndices) yield new rowSlice(i)
  def columns: Iterable[columnSlice] = for (j <- columnIndices) yield new columnSlice(j)

/*  def columns: Iterable[Array[T]] = for (j <- columnIndices) yield columnSlice(j)*/

  def iterator: Iterator[T] = a.iterator
}

object MultiArray2 {
  def fill[T](n1: Int, n2: Int)(elem: => T)(implicit tct: ClassTag[T]): MultiArray2[T] =
    new MultiArray2[T](n1, n2, Array.fill[T](n1 * n2)(elem))
}


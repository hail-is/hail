package org.apache.spark

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.spark.util.CollectionsUtils

import scala.reflect.ClassTag

case class OrderedPartitioner[T, K](
  rangeBounds: Array[T],
  projectKey: (K) => T,
  ascending: Boolean = true)(implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T], kct: ClassTag[K])
  extends Partitioner {
  import Ordering.Implicits._

  require(rangeBounds.isEmpty ||
    rangeBounds.zip(rangeBounds.tail).forall { case (left, right) => left < right })

  def write(out: ObjectOutputStream) {
    out.writeBoolean(ascending)
    out.writeObject(rangeBounds)
  }

  def numPartitions: Int = rangeBounds.length + 1

  var binarySearch: ((Array[T], T) => Int) = CollectionsUtils.makeBinarySearch[T]

  def getPartition(key: Any): Int = getPartitionT(projectKey(key.asInstanceOf[K]))

  /**
    * Code mostly copied from:
    *   org.apache.spark.RangePartitioner.getPartition(key: Any)
    *   version 1.5.0
    **/
  def getPartitionT(key: T): Int = {

    var partition = 0
    if (rangeBounds.length <= 128) {
      // If we have less than 128 partitions naive search
      while (partition < rangeBounds.length && key > rangeBounds(partition)) {
        partition += 1
      }
    } else {
      // Determine which binary search method to use only once.
      partition = binarySearch(rangeBounds, key)
      // binarySearch either returns the match location or -[insertion point]-1
      if (partition < 0) {
        partition = -partition - 1
      }
      if (partition > rangeBounds.length) {
        partition = rangeBounds.length
      }
    }
    if (ascending) {
      partition
    } else {
      rangeBounds.length - partition
    }
  }

  override def equals(other: Any): Boolean = other match {
    case r: OrderedPartitioner[_, _] => r.rangeBounds.sameElements(rangeBounds) && r.ascending == ascending
    case _ => false
  }

  override def hashCode(): Int = {
    val prime = 31
    var result = 1
    var i = 0
    while (i < rangeBounds.length) {
      result = prime * result + rangeBounds(i).hashCode
      i += 1
    }
    result = prime * result + ascending.hashCode
    result
  }

  def mapMonotonic[K2](newF: (K2) => T)(implicit k2Ord: Ordering[K2], k2ct: ClassTag[K2]): OrderedPartitioner[T, K2] = {
    new OrderedPartitioner[T, K2](rangeBounds, newF, ascending)
  }
}

object OrderedPartitioner {
  def empty[T, K](projectKey: (K) => T)(implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T],
    kct: ClassTag[K]): OrderedPartitioner[T, K] = new OrderedPartitioner(Array.empty[T], projectKey)

  def read[T, K](in: ObjectInputStream, projectKey: (K) => T)(implicit tOrd: Ordering[T], kOrd: Ordering[K], tct: ClassTag[T],
    kct: ClassTag[K]): OrderedPartitioner[T, K] = {
    val ascending = in.readBoolean()
    val rangeBounds = in.readObject().asInstanceOf[Array[T]]
    OrderedPartitioner(rangeBounds, projectKey, ascending)
  }
}

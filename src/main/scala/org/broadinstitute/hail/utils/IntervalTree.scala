package org.broadinstitute.hail.utils

import org.broadinstitute.hail.check._

import scala.collection.mutable
import scala.math.Ordering.Implicits._

// interval inclusive of start, exclusive of end: [start, end)
case class Interval[T](start: T, end: T)(implicit ev: Ordering[T]) extends Ordered[Interval[T]] {

  import ev._

  require(start <= end)

  def contains(position: T): Boolean = position >= start && position < end

  def isEmpty: Boolean = start == end

  def compare(that: Interval[T]): Int = {
    var c = ev.compare(start, that.start)
    if (c != 0)
      return c

    ev.compare(end, that.end)
  }
}

object Interval {
  def gen[T: Ordering](tgen: Gen[T]): Gen[Interval[T]] =
    Gen.zip(tgen, tgen)
      .map { case (x, y) =>
        if (x < y)
          Interval(x, y)
        else
          Interval(y, x)
      }
}

case class IntervalTree[T: Ordering](root: Option[IntervalTreeNode[T]]) extends Traversable[Interval[T]] with Serializable {
  def contains(position: T): Boolean = root.exists(_.contains(position))

  def query(position: T): Set[Interval[T]] = {
    val b = Set.newBuilder[Interval[T]]
    root.foreach(_.query(b, position))
    b.result()
  }

  def foreach[U](f: (Interval[T]) => U) =
    root.foreach(_.foreach(f))
}

object IntervalTree {
  def apply[T: Ordering](intervals: Array[Interval[T]]): IntervalTree[T] =
    new IntervalTree[T](fromSorted(intervals.sorted, 0, intervals.length))

  def fromSorted[T: Ordering](intervals: Array[Interval[T]], start: Int, end: Int): Option[IntervalTreeNode[T]] = {
    if (start >= end)
      None
    else {
      val mid = (start + end) / 2
      val i = intervals(mid)
      val lft = fromSorted(intervals, start, mid)
      val rt = fromSorted(intervals, mid + 1, end)
      Some(IntervalTreeNode(i, lft, rt, {
        val max1 = lft.map(_.maximum.max(i.end)).getOrElse(i.end)
        rt.map(_.maximum.max(max1)).getOrElse(max1)
      }))
    }
  }

  def gen[T: Ordering](tgen: Gen[T]): Gen[IntervalTree[T]] = {
    Gen.buildableOf[Array[Interval[T]], Interval[T]](Interval.gen(tgen))
      .map(intervals => IntervalTree(intervals))
  }
}

case class IntervalTreeNode[T: Ordering](i: Interval[T],
  left: Option[IntervalTreeNode[T]],
  right: Option[IntervalTreeNode[T]],
  maximum: T) extends Traversable[Interval[T]] {

  def contains(position: T): Boolean = {
    position <= maximum &&
      (left.exists(_.contains(position)) ||
        (position >= i.start &&
          (i.contains(position) ||
            right.exists(_.contains(position)))))
  }

  def query(b: mutable.Builder[Interval[T], _], position: T) {
    if (position <= maximum) {
      left.foreach(_.query(b, position))
      if (position >= i.start) {
        right.foreach(_.query(b, position))
        if (i.contains(position))
          b += i
      }
    }
  }

  def foreach[U](f: (Interval[T]) => U) {
    left.foreach(_.foreach(f))
    f(i)
    right.foreach(_.foreach(f))
  }
}

package is.hail.utils

import is.hail.check._
import org.json4s.JValue
import org.json4s.JsonAST.JObject

import scala.collection.mutable
import scala.language.implicitConversions
import scala.math.Ordering.Implicits._
import scala.reflect.ClassTag

// interval inclusive of start, exclusive of end: [start, end)
case class Interval[T](start: T, end: T)(implicit ev: Ordering[T]) extends Ordered[Interval[T]] {

  import ev._

  require(start < end, s"invalid interval: $this: start is not before end")

  def contains(position: T): Boolean = position >= start && position < end

  def overlaps(other: Interval[T]): Boolean = this.contains(other.start) || other.contains(this.start)

  def isEmpty: Boolean = start == end

  def compare(that: Interval[T]): Int = {
    var c = ev.compare(start, that.start)
    if (c != 0)
      return c

    ev.compare(end, that.end)
  }

  def toJSON(f: (T) => JValue): JValue = JObject("start" -> f(start), "end" -> f(end))

  override def toString: String = start + "-" + end
}

object Interval {
  implicit def intervalOrder[T](ev: Ordering[T]): Ordering[Interval[T]] = new Ordering[Interval[T]] {
    def compare(x: Interval[T], y: Interval[T]): Int = x.compare(y)
  }

  def gen[T: Ordering](tgen: Gen[T]): Gen[Interval[T]] =
    Gen.zip(tgen, tgen)
      .filter { case (x, y) => x != y }
      .map { case (x, y) =>
        if (x < y)
          Interval(x, y)
        else
          Interval(y, x)
      }
}

case class IntervalTree[T: Ordering, U: ClassTag](root: Option[IntervalTreeNode[T, U]]) extends
  Traversable[(Interval[T], U)] with Serializable {
  def contains(position: T): Boolean = root.exists(_.contains(position))

  def overlaps(interval: Interval[T]): Boolean = root.exists(_.overlaps(interval))

  def queryIntervals(position: T): Array[Interval[T]] = {
    val b = Array.newBuilder[Interval[T]]
    root.foreach(_.query(b, position))
    b.result()
  }

  def queryValues(position: T): Array[U] = {
    val b = Array.newBuilder[U]
    root.foreach(_.queryValues(b, position))
    b.result()
  }

  def foreach[V](f: ((Interval[T], U)) => V) {
    root.foreach(_.foreach(f))
  }
}

object IntervalTree {
  def annotationTree[T: Ordering, U: ClassTag](values: Array[(Interval[T], U)]): IntervalTree[T, U] = {
    val sorted = values.sortBy(_._1)
    new IntervalTree[T, U](fromSorted(sorted, 0, sorted.length))
  }

  def apply[T: Ordering](intervals: Array[Interval[T]]): IntervalTree[T, Unit] = {
    val sorted = if (intervals.nonEmpty) {
      val unpruned = intervals.sorted
      val ab = new ArrayBuilder[Interval[T]](intervals.length)
      var tmp = unpruned.head
      var i = 1
      var pruned = 0
      while (i < unpruned.length) {
        val interval = unpruned(i)
        if (interval.start <= tmp.end) {
          val max = if (interval.end > tmp.end)
            interval.end
          else
            tmp.end
          tmp = Interval(tmp.start, max)
          pruned += 1
        } else {
          ab += tmp
          tmp = interval
        }

        i += 1
      }
      ab += tmp

      ab.result()
    } else intervals

    new IntervalTree[T, Unit](fromSorted(sorted.map(i => (i, ())), 0, sorted.length))
  }

  def fromSorted[T: Ordering, U](intervals: Array[(Interval[T], U)], start: Int, end: Int): Option[IntervalTreeNode[T, U]] = {
    if (start >= end)
      None
    else {
      val mid = (start + end) / 2
      val (i, v) = intervals(mid)
      val lft = fromSorted(intervals, start, mid)
      val rt = fromSorted(intervals, mid + 1, end)
      Some(IntervalTreeNode(i, lft, rt, {
        val max1 = lft.map(_.maximum.max(i.end)).getOrElse(i.end)
        rt.map(_.maximum.max(max1)).getOrElse(max1)
      }, v))
    }
  }

  def gen[T: Ordering](tgen: Gen[T]): Gen[IntervalTree[T, Unit]] = {
    Gen.buildableOf[Array, Interval[T]](Interval.gen(tgen)).map(IntervalTree.apply(_))
  }
}

case class IntervalTreeNode[T: Ordering, U](i: Interval[T],
  left: Option[IntervalTreeNode[T, U]],
  right: Option[IntervalTreeNode[T, U]],
  maximum: T, value: U) extends Traversable[(Interval[T], U)] {

  def contains(position: T): Boolean = {
    position <= maximum &&
      (left.exists(_.contains(position)) ||
        (position >= i.start &&
          (i.contains(position) ||
            right.exists(_.contains(position)))))
  }

  def overlaps(interval: Interval[T]): Boolean = {
    interval.start <= maximum && (left.exists(_.overlaps(interval))) ||
      i.overlaps(interval) || (right.exists(_.overlaps(interval)))
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

  def queryValues(b: mutable.Builder[U, _], position: T) {
    if (position <= maximum) {
      left.foreach(_.queryValues(b, position))
      if (position >= i.start) {
        right.foreach(_.queryValues(b, position))
        if (i.contains(position))
          b += value
      }
    }
  }

  def foreach[V](f: ((Interval[T], U)) => V) {
    left.foreach(_.foreach(f))
    f((i, value))
    right.foreach(_.foreach(f))
  }
}

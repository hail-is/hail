package is.hail.utils

import is.hail.annotations.ExtendedOrdering
import is.hail.check._
import is.hail.expr.types.TBoolean
import org.json4s.JValue
import org.json4s.JsonAST.JObject

import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

case class Interval(start: Any, end: Any, includeStart: Boolean, includeEnd: Boolean) extends Serializable {

  def contains(pord: ExtendedOrdering, position: Any): Boolean = {
    val compareStart = pord.compare(position, start)
    val compareEnd = pord.compare(position, end)
    (compareStart > 0 || (includeStart && compareStart == 0)) && (compareEnd < 0 || (includeEnd && compareEnd == 0))
  }

  private def matchesEndpoint(pord: ExtendedOrdering, position: Any): Boolean = {
    pord.eq(position, start) || pord.eq(position, end)
  }

  def overlaps(pord: ExtendedOrdering, other: Interval): Boolean = {
    (this.contains(pord, other.start) && (other.includeStart || !pord.equiv(this.end, other.start))) ||
      (other.contains(pord, this.start) && (this.includeStart || !pord.equiv(other.end, this.start)))
  }

  // true indicates definitely-empty interval, but false does not guarantee
  // non-empty interval in the (a, b) case;
  // e.g. (1,2) is an empty Interval(Int32), but we cannot guarantee distance
  // like that right now.
  def isEmpty(pord: ExtendedOrdering): Boolean = if (includeStart && includeEnd) pord.gt(start, end) else pord.gteq(start, end)

  def toJSON(f: (Any) => JValue): JValue =
    JObject("start" -> f(start),
      "end" -> f(end),
      "includeStart" -> TBoolean().toJSON(includeStart),
      "includeEnd" -> TBoolean().toJSON(includeEnd))


  override def toString: String = (if (includeStart) "[" else "(") + start + "-" + end + (if (includeEnd) "]" else ")")
}

object Interval {
  def gen[P](pord: ExtendedOrdering, pgen: Gen[P]): Gen[Interval] =
    Gen.zip(pgen, pgen, Gen.coin(), Gen.coin())
      .map { case (x, y, s, e) =>
        if (pord.lt(x, y))
          Interval(x, y, s, e)
        else
          Interval(y, x, s, e)
      }

  def ordering(pord: ExtendedOrdering): ExtendedOrdering = new ExtendedOrdering {
    def compareNonnull(x: Any, y: Any, missingGreatest: Boolean): Int = {
      val xi = x.asInstanceOf[Interval]
      val yi = y.asInstanceOf[Interval]

      val c = pord.compare(xi.start, yi.start, missingGreatest)
      if (c != 0)
        return c
      if (xi.includeStart != yi.includeStart)
        return if (xi.includeStart) -1 else 1

      val c2 = pord.compare(xi.end, yi.end, missingGreatest)
      if (c2 != 0)
        return c2
      if (xi.includeEnd != yi.includeEnd)
        if (xi.includeEnd) 1 else -1
      else 0
    }
  }
}

case class IntervalTree[U: ClassTag](root: Option[IntervalTreeNode[U]]) extends
  Traversable[(Interval, U)] with Serializable {
  def contains(pord: ExtendedOrdering, position: Any): Boolean = root.exists(_.contains(pord, position))

  def overlaps(pord: ExtendedOrdering, interval: Interval): Boolean = root.exists(_.overlaps(pord, interval))

  def queryIntervals(pord: ExtendedOrdering, position: Any): Array[Interval] = {
    val b = Array.newBuilder[Interval]
    root.foreach(_.query(pord, b, position))
    b.result()
  }

  def queryValues(pord: ExtendedOrdering, position: Any): Array[U] = {
    val b = Array.newBuilder[U]
    root.foreach(_.queryValues(pord, b, position))
    b.result()
  }

  def foreach[V](f: ((Interval, U)) => V) {
    root.foreach(_.foreach(f))
  }
}

object IntervalTree {
  def annotationTree[U: ClassTag](pord: ExtendedOrdering, values: Array[(Interval, U)]): IntervalTree[U] = {
    val iord = Interval.ordering(pord)
    val sorted = values.sortBy(_._1)(iord.toOrdering.asInstanceOf[Ordering[Interval]])
    new IntervalTree[U](fromSorted(pord, sorted, 0, sorted.length))
  }

  def apply[T](pord: ExtendedOrdering, intervals: Array[Interval]): IntervalTree[Unit] = {
    val iord = Interval.ordering(pord)
    val sorted = if (intervals.nonEmpty) {
      val unpruned = intervals.sorted(iord.toOrdering.asInstanceOf[Ordering[Interval]])
      val ab = new ArrayBuilder[Interval](intervals.length)
      var tmp = unpruned.head
      var i = 1
      var pruned = 0
      while (i < unpruned.length) {
        val interval = unpruned(i)
        val c = pord.compare(interval.start, tmp.end)
        if (c < 0 || (c == 0 && (interval.includeStart || tmp.includeEnd))) {
          tmp = if (pord.lt(interval.end, tmp.end))
            Interval(tmp.start, tmp.end, tmp.includeStart, tmp.includeEnd)
          else if (pord.equiv(interval.end, tmp.end))
            Interval(tmp.start, tmp.end, tmp.includeStart, tmp.includeEnd || interval.includeEnd)
          else
            Interval(tmp.start, interval.end, tmp.includeStart, interval.includeEnd)
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

    new IntervalTree[Unit](fromSorted(pord, sorted.map(i => (i, ())), 0, sorted.length))
  }

  def fromSorted[U](pord: ExtendedOrdering, intervals: Array[(Interval, U)], start: Int, end: Int): Option[IntervalTreeNode[U]] = {
    if (start >= end)
      None
    else {
      val mid = (start + end) / 2
      val (i, v) = intervals(mid)
      val lft = fromSorted(pord, intervals, start, mid)
      val rt = fromSorted(pord, intervals, mid + 1, end)
      Some(IntervalTreeNode(i, lft, rt, {
        val min1 = lft.map(x => pord.min(x.minimum, i.start)).getOrElse(i.start)
        rt.map(x => pord.min(x.minimum, min1)).getOrElse(min1)
      },
        {
        val max1 = lft.map(x => pord.max(x.maximum, i.end)).getOrElse(i.end)
        rt.map(x => pord.max(x.maximum, max1)).getOrElse(max1)
      }, v))
    }
  }

  def gen[T](pord: ExtendedOrdering, pgen: Gen[T]): Gen[IntervalTree[Unit]] = {
    Gen.buildableOf[Array](Interval.gen(pord, pgen)).map(a => IntervalTree.apply(pord, a))
  }
}

case class IntervalTreeNode[U](i: Interval,
  left: Option[IntervalTreeNode[U]],
  right: Option[IntervalTreeNode[U]],
  minimum: Any, maximum: Any, value: U) extends Traversable[(Interval, U)] {

  def contains(pord: ExtendedOrdering, position: Any): Boolean = {
    pord.gteq(position, minimum) && pord.lteq(position, maximum) &&
      (left.exists(_.contains(pord, position)) ||
        (pord.gteq(position, i.start) &&
          (i.contains(pord, position) ||
            right.exists(_.contains(pord, position)))))
  }

  def overlaps(pord: ExtendedOrdering, interval: Interval): Boolean = {
    pord.gteq(interval.end, minimum) && pord.lteq(interval.start, maximum) &&
      (left.exists(_.overlaps(pord, interval))) ||
      i.overlaps(pord, interval) || (right.exists(_.overlaps(pord, interval)))
  }

  def query(pord: ExtendedOrdering, b: mutable.Builder[Interval, _], position: Any) {
    if (pord.gteq(position, minimum) && pord.lteq(position, maximum)) {
      left.foreach(_.query(pord, b, position))
      if (pord.gteq(position, i.start)) {
        right.foreach(_.query(pord, b, position))
        if (i.contains(pord, position))
          b += i
      }
    }
  }

  def queryValues(pord: ExtendedOrdering, b: mutable.Builder[U, _], position: Any) {
    if (pord.gteq(position, minimum) && pord.lteq(position, maximum)) {
      left.foreach(_.queryValues(pord, b, position))
      if (pord.gteq(position, i.start)) {
        right.foreach(_.queryValues(pord, b, position))
        if (i.contains(pord, position))
          b += value
      }
    }
  }

  def foreach[V](f: ((Interval, U)) => V) {
    left.foreach(_.foreach(f))
    f((i, value))
    right.foreach(_.foreach(f))
  }
}
